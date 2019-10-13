"""
@author:lei
blog:https://leigang431.github.io/
mooc:深度学习应用开发-TensorFlow实践
aim:单作用变量的线性回归
"""
# 导入第三方库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 准备数据集
np.random.seed(5)
x_data = np.linspace(-1, 1, 100)
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4
# plt.scatter(x_data,y_data)
# plt.plot(x_data,2.0*x_data+1.0,color='red',linewidth=3)
# plt.show()

# 构建模型
##定义训练数据的占位符，x是特征值，y是标签值
x = tf.placeholder("float", name="x")
y = tf.placeholder("float", name="y")


##定义模型
def model(x, w, b):
    return tf.multiply(x, w) + b


##定义模型结构
w = tf.Variable(1.0, name="w0")
b = tf.Variable(0.0, name="b0")
pred = model(x, w, b)

# 训练模型
##设置训练参数
# 迭代次数
train_epochs = 10
# 学习率
learning_rate = 0.05

# 定义损失函数
loss_function = tf.reduce_mean(tf.square(y - pred))
# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
# 创建会话
# 定义会话
sess = tf.Session()
writer = tf.summary.FileWriter("E:/TensorBoard/test", sess.graph)
# 变量初始化
init = tf.global_variables_initializer()
sess.run(init)
step=0
loss_list=[]
# 迭代训练
for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})
        display_step=10
        loss_list.append(loss)
        step+=1
        if step%display_step == 0:
            print("train epoch:",'%02d'%(epoch+1),"step:",'%03d'%(step),"loss=",":%9f"%(loss))
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    #plt.plot(x_data, w0temp * x_data + b0temp)  # 画图
plt.figure(1)
plt.plot(loss_list)
plt.figure(2)
plt.plot(loss_list,'r+')
plt.show()
# 打印结果
print("w:", sess.run(w))
print("b:", sess.run(b))
# 结果可视化
# plt.scatter(x_data, y_data, label='original data')
# plt.plot(x_data, x_data * sess.run(w) + sess.run(b), \
#          label='fitted line', color='red', linewidth=3)
# plt.legend(loc=2)
# plt.show()

# 使用模型，进行预测
x_test = 3.21
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值：%f" % predict)
target = 2.0 * x_test + 1.0
print("目标值：%f" % target)
writer.close()