import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BS_curve_torch(object):
    """
    PyTorch implementation of B-spline curve with GPU support and automatic differentiation.
    """
    
    def __init__(self, n, p, cp=None, knots=None, device='cpu', dtype=torch.float32):
        """
        Initialize B-spline curve.
        
        Args:
            n: Number index (n+1 control points: p0, p1, ..., pn)
            p: Degree of B-spline
            cp: Control points tensor
            knots: Knot vector tensor
            device: Computing device ('cpu', 'cuda')
            dtype: Data type for tensors
        """
        self.n = n  # n+1 control points >>> p0,p1,,,pn
        self.p = p
        self.m = n + p + 1  # m+1 knots >>> u0,u1,,,um

        self.paras = None
        self.u = None
        self.cp = None
        
        self.device = device
        self.dtype = dtype


        self.paras = None

    def coeffs(self, uq):
        """
        Compute B-spline coefficients using recursive formula.
        
        Args:
            uq: Parameter value
            
        Returns:
            B-spline basis functions
        """
        # Convert to tensor if needed
        uq = torch.as_tensor(uq, dtype=self.dtype, device=self.device)
        
        # N[] holds all intermediate and final results
        N = torch.zeros(self.m + 1, dtype=self.dtype, device=self.device)
       
        # Handle special cases - Important Properties of clamped B-spline curve
        if torch.isclose(uq, self.u[0]):
            N = N.clone()
            N[0] = 1.0
            return N[0:self.n + 1]
        elif torch.isclose(uq, self.u[self.m]):
            N = N.clone()
            N[self.n] = 1.0
            return N[0:self.n + 1]

        # Find knot span k where uq is in [uk, uk+1)
        check = uq - self.u
        ind = check >= 0
        nonzero_indices = torch.nonzero(ind, as_tuple=False)
        if nonzero_indices.numel() == 0:
            # If no indices are found, uq is before the first knot
            k = 0
        else:
            k = torch.max(nonzero_indices).item()

        N = N.clone()
        N[k] = 1.0  # degree 0

        # Degree d goes from 1 to p
        for d in range(1, self.p + 1):
            r_max = self.m - d - 1  # maximum subscript value of N in degree d
            N = N.clone()  # 创建副本避免就地操作

            if k - d >= 0:
                denominator = self.u[k + 1] - self.u[k - d + 1]
                if denominator != 0:
                    N_new = N.clone()
                    N_new[k - d] = (self.u[k + 1] - uq) / denominator * N[k - d + 1]
                    N = N_new
                else:
                    N_new = N.clone()
                    N_new[k - d] = (self.u[k + 1] - uq) * N[k - d + 1]
                    N = N_new

            for i in range(k - d + 1, k):
                if i >= 0 and i <= r_max:
                    Denominator1 = self.u[i + d] - self.u[i]
                    Denominator2 = self.u[i + d + 1] - self.u[i + 1]
                    
                    # Handle 0/0 = 0 cases
                    if Denominator1 == 0:
                        term1 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                    else:
                        term1 = (uq - self.u[i]) / Denominator1 * N[i]
                    
                    if Denominator2 == 0:
                        term2 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                    else:
                        term2 = (self.u[i + d + 1] - uq) / Denominator2 * N[i + 1]
                    
                    N_new = N.clone()
                    N_new[i] = term1 + term2
                    N = N_new

            if k <= r_max:
                denominator = self.u[k + d] - self.u[k]
                if denominator != 0:
                    N_new = N.clone()
                    N_new[k] = (uq - self.u[k]) / denominator * N[k]
                    N = N_new
                else:
                    N_new = N.clone()
                    N_new[k] = (uq - self.u[k]) * N[k]
                    N = N_new

        return N[0:self.n + 1]

    def De_Boor(self, uq):
        """
        Calculate point coordinates using De Boor's algorithm.
        
        Args:
            uq: Parameter value
            
        Returns:
            Point on the curve
        """
        uq = torch.as_tensor(uq, dtype=self.dtype, device=self.device)
        
        # Find knot span k where uq is in [uk, uk+1)
        # check = uq - self.u
        
        check = uq - self.u
        ind = check >= 0
        nonzero_indices = torch.nonzero(ind, as_tuple=False)
        if nonzero_indices.numel() == 0:
            # If no indices are found, uq is before the first knot
            k = 0
        else:
            k = torch.max(nonzero_indices).item()
        
        # Calculate multiplicity and insertion count
        if torch.any(torch.isclose(uq, self.u)):
            # Find multiplicity of u[k]
            sk = torch.sum(torch.isclose(self.u, self.u[k])).item()
            h = self.p - sk
        else:
            sk = 0
            h = self.p
        
        # Handle special cases
        if h == -1:
            if k == self.p:
                return self.cp[0].clone()
            elif k == self.m:
                return self.cp[-1].clone()

        # Initial values of P (affected control points)
        P = self.cp[k - self.p:k - sk + 1].clone()
        
        # De Boor's algorithm
        for r in range(1, h + 1):
            temp = []
            for i in range(k - self.p + r, k - sk + 1):
                a_ir = (uq - self.u[i]) / (self.u[i + self.p - r + 1] - self.u[i])
                temp.append((1 - a_ir) * P[i - (k - self.p) - 1] + a_ir * P[i - (k - self.p)])
            
            if temp:
                temp_tensor = torch.stack(temp)
                start_idx = k - self.p + r - (k - self.p)
                end_idx = k - sk + 1 - (k - self.p)
                # 避免就地操作，创建新的张量
                P_new = P.clone()
                P_new[start_idx:end_idx] = temp_tensor
                P = P_new

        return P[-1]

    def bs(self, us):
        """
        Calculate curve points using De Boor algorithm.
        
        Args:
            us: Parameter values tensor or array
            
        Returns:
            Points on the curve
        """
        us = torch.as_tensor(us, dtype=self.dtype, device=self.device)
        y = []
        for u in us:
            y.append(self.De_Boor(u))
        return torch.stack(y)

    def estimate_parameters(self, data_points):
        """
        Estimate parameters using centripetal method.
        
        Args:
            data_points: Data points tensor
            method: Parameterization method
            
        Returns:
            Parameter values
        """
        pts = torch.as_tensor(data_points, dtype=self.dtype, device=self.device)
        N = pts.shape[0]
        w = pts.shape[1]
        
        # Calculate chord lengths
        Li = []
        for i in range(1, N):
            chord_length = torch.norm(pts[i] - pts[i-1])
            Li.append(chord_length)
        
        Li = torch.stack(Li)
        L = torch.sum(Li)

        # Calculate cumulative parameters
        t = [torch.tensor(0.0, dtype=self.dtype, device=self.device)]
        for i in range(len(Li)):
            Lki = torch.sum(Li[:i+1])
            t.append(Lki / L)
        
        t = torch.stack(t)
    
        self.paras = t
        
        # # Clamp values to [0, 1]
        # t = torch.clamp(t, 0.0, 1.0)
        return t

    def get_knots(self):
        """
        Generate knot vector using averaging method.
        
        Args:
            method: Knot generation method
            
        Returns:
            Knot vector
        """
        # Initialize knots with p+1 zeros
        knots = torch.zeros(self.p + 1, dtype=self.dtype, device=self.device)
        

        # paras_temp = self.paras
        
        # Select parameters for averaging
        num = self.m - self.p  # select n+1 parameters
        
        indices = torch.linspace(0, self.paras.shape[0] - 1, num, dtype=torch.long)
        paras_knots = self.paras[indices]

        # Calculate interior knots using averaging
        interior_knots = []
        for j in range(1, self.n - self.p + 1):
            k_temp = torch.mean(paras_knots[j:j + self.p])
            interior_knots.append(k_temp)
        
        if interior_knots:
            interior_knots = torch.stack(interior_knots)
            knots = torch.cat([knots, interior_knots])
        
        # Add p+1 ones at the end
        ones = torch.ones(self.p + 1, dtype=self.dtype, device=self.device)
        knots = torch.cat([knots, ones])
        
        self.u = knots

        return knots

    def approximation(self, pts):
        """
        Generate control points using least squares approximation.
        
        Args:
            pts: Data points tensor
            
        Returns:
            Control points
        """
        pts = torch.as_tensor(pts, dtype=self.dtype, device=self.device)
        num = pts.shape[0] - 1  # (num+1) is the number of data points

        # Initialize control points
        P = torch.zeros((self.n + 1, pts.shape[1]), dtype=self.dtype, device=self.device)
        P = P.clone()  # 创建副本避免就地操作
        P[0] = pts[0]
        P[-1] = pts[-1]

        # Compute basis function matrix N
        N = []
        # paras = self.paras
        for uq in self.paras:
            N_temp = self.coeffs(uq)
            N.append(N_temp)
        N = torch.stack(N)

        # Compute Q vector
        Q = [torch.zeros_like(pts[0])]  # placeholder for index 0
        for k in range(1, num):
            Q_temp = pts[k] - N[k, 0] * pts[0] - N[k, self.n] * pts[-1]
            Q.append(Q_temp)

        # Compute b vector
        b = []
        for i in range(1, self.n):
            b_temp = torch.zeros_like(pts[0])
            for k in range(1, num):
                b_temp += N[k, i] * Q[k]
            b.append(b_temp)
        
        if b:
            b = torch.stack(b)
            
            # Extract interior basis functions
            N_interior = N[:, 1:self.n]
            
            # Solve linear system A * P_interior = b
            A = torch.mm(N_interior.T, N_interior)
            
            # Check if A is well-conditioned
            if A.shape[0] == 0 or A.shape[1] == 0:
                # If no interior control points, return original P
                return P.clone()
            
            # Solve for each dimension separately
            cpm = []
            for dim in range(pts.shape[1]):
                b_dim = b[:, dim]
                try:
                    # Try to solve the linear system
                    cpm_dim = torch.linalg.solve(A, b_dim)
                except torch._C._LinAlgError:
                    # If the matrix is singular, use pseudo-inverse with regularization
                    # Add small regularization term to diagonal
                    A_reg = A + torch.eye(A.shape[0], dtype=A.dtype, device=A.device) * 1e-6
                    try:
                        cpm_dim = torch.linalg.solve(A_reg, b_dim)
                    except torch._C._LinAlgError:
                        # If still singular, use pseudo-inverse
                        cpm_dim = torch.linalg.pinv(A) @ b_dim
                cpm.append(cpm_dim)
            
            cpm = torch.stack(cpm, dim=1)
            P_new = P.clone()  # 创建副本
            P_new[1:self.n] = cpm
            P = P_new

        self.cp = P.clone()
        return P.clone()


    def to(self, device):
        """Move all tensors to specified device."""
        self.device = torch.device(device)
        if self.cp is not None:
            self.cp = self.cp.to(self.device)
        if self.u is not None:
            self.u = self.u.to(self.device)
        if self.paras is not None:
            self.paras = self.paras.to(self.device)
        return self


def demo_with_numpy_data():
    """Demo using numpy data (converted to torch internally)."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create B-spline curve
    bs = BS_curve_torch(9, 3, device=device)
    
    # Generate test data with numpy
    xx = np.linspace(0, 4 * np.pi, 101)
    yy = np.sin(xx) + 0.6 * np.random.random(101)
    data_np = np.array([xx, yy]).T
    
    # Convert to torch tensor
    data_torch = torch.tensor(data_np, dtype=torch.float64, device=device)
    
    print("Demo with numpy data converted to torch:")
    plot_bspline_result(bs, data_torch, "PyTorch B-spline (from NumPy data)")

def demo_with_torch_data():
    """Demo using native torch data."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create B-spline curve
    bs = BS_curve_torch(9, 3, device=device)
    
    # Generate test data directly with torch
    xx = torch.linspace(0, 4 * torch.pi, 101, device=device, dtype=torch.float64)
    noise = torch.rand(101, device=device, dtype=torch.float64) * 0.6
    yy = torch.sin(xx) + noise
    
    # Stack to create data tensor [N, 2]
    data_torch = torch.stack([xx, yy], dim=1)
    
    print("Demo with native torch data:")
    plot_bspline_result(bs, data_torch, "PyTorch B-spline (native torch data)")

def plot_bspline_result(bs, data_torch, title):
    """Helper function to fit curve and plot results."""
    print(f"Input data shape: {data_torch.shape}, device: {data_torch.device}")
    
    # Fit B-spline curve (all operations stay in torch)
    paras = bs.estimate_parameters(data_torch)
    knots = bs.get_knots()
    print(f"Knots shape: {knots.shape}, device: {knots.device}")
    
    # if bs.check():
    cp = bs.approximation(data_torch)
    print(f"Control points shape: {cp.shape}, device: {cp.device}")

    # Generate curve points
    uq = torch.linspace(0, 1, 101, device=data_torch.device, dtype=data_torch.dtype)
    y = bs.bs(uq)
    print(f"Curve points shape: {y.shape}, device: {y.device}")
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Convert to numpy only for plotting
    data_np = data_torch.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    cp_np = cp.detach().cpu().numpy()
    
    ax.scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, label='Data Points', color='gray')
    ax.plot(y_np[:, 0], y_np[:, 1], '-r', linewidth=2, label='B-spline Curve')
    ax.plot(cp_np[:, 0], cp_np[:, 1], '--b*', markersize=8, label='Control Points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    # else:
    #     print("B-spline validation failed")

if __name__ == "__main__":
    # Demo 1: Using numpy data (common case)
    demo_with_numpy_data()
    
    # Demo 2: Using native torch data (preferred for torch workflows)
    demo_with_torch_data()