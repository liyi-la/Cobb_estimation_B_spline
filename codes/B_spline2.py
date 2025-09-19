import numpy as np
import matplotlib.pyplot as plt

# 计算在某一特定t下的 B_{i,k}
def getBt(controlPoints, knots, t):
	# calculate m,n,k
	m = knots.shape[0]-1
	n = controlPoints.shape[0]-1
	k = m - n - 1
	# initialize B by zeros 
	B = np.zeros((k+1, m))

	# get t region
	tStart = 0
	for x in range(m+1):
		if t==1:
			tStart = m-1
		if knots[x] > t:
			tStart = x-1
			break
	 
	# calculate B(t)
	for _k in range(k+1):
		if _k == 0:
			B[_k, tStart] = 1
		else:
			for i in range(m-_k):
				if knots[i+_k]-knots[i]== 0:
					w1 = 0
				else:
					w1 = (t-knots[i])/(knots[i+_k]-knots[i]) 
				if knots[i+_k+1]-knots[i+1] == 0:
					w2 = 0
				else:
					w2 = (knots[i+_k+1]-t)/(knots[i+_k+1]-knots[i+1])
				B[_k,i] = w1*B[_k-1, i] + w2*B[_k-1, i+1]
	return B

# 绘制 B_{i,k}(t)函数
def plotBt(Bt,num, i,k):
	print(k,i)
	Bt = np.array(Bt)
	tt = np.linspace(0,1,num)
	yy = [Bt[t,k,i] for t in range(num)]
	plt.plot(tt, yy)

# 根据最后一列（最高阶次）的 B(t)，即权重，乘以控制点坐标，从而求出曲线上点坐标
def getPt(Bt, controlPoints):
	Bt = np.array(Bt)
	ptArray = Bt.reshape(-1,1) * controlPoints
	pt = ptArray.sum(axis = 0)
	return pt

# 绘制出生成的样条曲线: useReg 表示是否使用曲线有效定义域[t_k, t_{m-k}]
def main1(useReg = True):
    controlPoints = np.array([[50,50], [100,300], [300,100], [380,200], [400,300],[450,350], [500,210],[600,140],[650,400],[680,300]])
    knots = np.array([0,0,0,0,2/9,2/9,3/9,4/9,5/9,7/9,1,1,1,1])
    m = knots.shape[0]-1
    n = controlPoints.shape[0]-1
    k = m - n - 1
    print('n:',n)
    print('m:',m)
    print('k:',k)
    
    for t in np.linspace(0,1,100):
        if useReg and not(t >= knots[k] and t<= knots[n+1]):
            continue
        Bt = getBt(controlPoints, knots, t)
        Pt = getPt(Bt[k, :n+1], controlPoints)
        plt.scatter(Pt[0],Pt[1],color='b')
    plt.scatter(controlPoints[:,0], controlPoints[:,1],color = 'r')
    plt.show()

# 绘制 B_{i,k} 变化图:如果不给定{i,k}则显示所有B{i,k}(t)图像
def main2(i=-1,k=-1):
	controlPoints = np.array([[50,50], [100,300], [300,100], [380,200], [400,600]])
	knots = np.array([0,1/9,2/9,3/9,4/9,5/9,6/9,7/9,8/9,1])
	m = knots.shape[0]-1
	n = controlPoints.shape[0]-1
	k = m - n - 1
	print('n:',n)
	print('m:',m)
	print('k:',k)
	B = []
	num = 100 # 离散点数目
	for t in np.linspace(0,1,num):
		Bt = getBt(controlPoints, knots, t)
		B.append(list(Bt))

	figure1 = plt.figure('B_{i,k}')
	if i==-1:
		fig = []
		for i in range(n+1):
			for k in range(k+1):
				plotBt(B,num, i,k)
				fig.append('B_{%d,%d}'%(i,k))
	else:
		plotBt(B,num, i,k)
		fig.append('B_{%d,%d}'%(i,k))
	plt.legend(fig)
	plt.show()   
    
if __name__ == '__main__':
    main1()
    main2()
