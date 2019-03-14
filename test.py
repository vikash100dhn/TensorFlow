import matplotlib.pyplot as plt
a=[[1,2,3],[4,5,6]]
img_data=list(zip(*a))
plt.imshow(img_data)
plt.show()