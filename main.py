import matplotlib.pyplot as plt
from prettytable import PrettyTable
import dotlis as dl
import neural as nl
import time
import sigma

# นำเข้าชุดข้อมูลที่ใช้สำหรับ train และ test
train = dl.read_csv("datasets/cancer_train.txt")
train.retype([float for i in range(9)] + [int])

test = dl.read_csv("datasets/cancer_test.txt")
test.retype([float for i in range(9)] + [int])

X_train, y_train = train.exclass("Class")
X_test, y_test = test.exclass("Class")

# สร้าง NN โดย set layer ไว้ 3 layer ได้แก่
# 1. input layer 9 node ตามจำนวน attribute ที่ใช้ 9 attributes
# 2. 1 hidden layer ที่มี 10 node
# 3. output layer 2 node ตามจำนวน class ที่ต้องทำนาย
# หมายเหตุ: ทุก node ใช้ sigmoid เป็น activation function
clf = nl.NeuralNetwork()
clf.setNetwork(
    [
        nl.Layer(9, sigma.Sigmoid()),
        nl.Layer(10, sigma.Sigmoid()),
        nl.Layer(2, sigma.Sigmoid())
    ]
)

# จับเวลาที่ใช้ในการ train ทั้งหมด
start_time = time.time()

clf.fit(X_train, y_train, 200, 0.03) # เริ่ม train NN โดยใช้เฉพาะ test data

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# แสดงผลลัพธ์ที่ได้
print("-------------------------------------------------")
# หา accuracy ของข้อมูลที่ใช้ train
predict_train = clf.predict(X_train)
print("train accuracy: " + str(sum(x == y for x, y in zip(predict_train, y_train.column(0)))/len(predict_train)))

# หา accuracy ของข้อมูลที่ใช้ test
predict_test = clf.predict(X_test)
print("test accuracy: " + str(sum(x == y for x, y in zip(predict_test, y_test.column(0)))/len(predict_test)))
print()

t = PrettyTable(["A"+str(i) for i in range(len(X_test.attributes))] + ["Class","predict"])
for i in range(len(test.rows)):
    result = clf.predict_one(X_test.rows[i])
    if result != y_test.column(0)[i]:
        t.add_row(test.rows[i] + [result])
print(t)

# plot line graph ของ error ที่เกิดขึ้นในแต่ละ epoch
plt.plot(clf.epoch_error, linestyle='-', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Neural Network Training Error Over Epochs')
plt.show()
