import random

# Class ของ Neural Network 
# ที่สามารถเปลี่ยนแปลงจำนวน node และ layer ได้
class NeuralNetwork:
    def __init__(self):
        self.X = None
        self.y = None
        self.classes = []
        self.inputLayer = [] # เก็บ nodes ของ input layer
        self.hiddenLayer_ls = [] # เก็บ layers ที่เป็น hidden layer
        self.outputLayer = [] # เก็บ nodes ของ output layer
        self.epoch_error = []
    
    # นำเข้าชุดข้อมูลที่ต้องการ train เข้ามา และทำการ train ตามจำนวนรอบ epoch และ lr
    def fit(self, X, y, epochs, lr=0.01):
        self.X = X
        self.y = y
        self.classes = list(set(y.column(0)))
        self.epoch_error = []

        # ใน 1 epoch จะมีการทำ forward, backprop และ ปรับ weight กับทุกๆ ข้อมูลที่รับมา
        for epoch in range(epochs):
            error_ls = []
            for i in range(len(self.X.rows)): # วนข้อมูลทั้งหมด
                input_ls = self.X.rows[i]

                # ทำการปรับ output จาก class เป็น list ของสิ่งที่แต่ละ output node ควรจะตอบ
                # เช่น ถ้ามี 2 class จำนวน output node จะมีอยู่ 2 nodes ทำให้ list ต้องมี 2 elements ด้วย
                # โดยที่ element แรกคือคำตอบที่ควรจะได้จาก node ที่ 1 และ element 2 สำหรับ node ที่ 2
                # ซึ่ง class 1 จะถูกแปลงเป็น [1,0] และ class 2 เป็น [0,1]
                output_ls = [int(y.column(0)[i] == c) for c in self.classes]

                # ทำกระบวนการ forward และหา error ของ output nodes ทั้งหมดเพิ่มไปใน list 
                # ของ error ที่เกิดขึ้นจากแต่ละข้อมูล
                self.__forward(input_ls)
                error_ls.append(self.__get_error(output_ls))

                # ทำกระบวนการ backpropagation และปรับ weight ตาม error ล่าสุดจาก output node 
                self.__backprop(error_ls[i], lr)

            # หา error รวมของแต่ละ epoch และแสดงผลออกมาหลังจาก epoch นั้นๆ เสร็จแล้ว
            sum_error = 0
            for i in range(len(error_ls)):
                sum_error += (1/2)*sum([err*err for err in error_ls[i]])
            avg_error = sum_error/len(error_ls)
            print(f"epoch {epoch}: avg_error = {avg_error}")
            self.epoch_error.append(avg_error)

    # forward ข้อมูลจาก input ที่ให้มาเพื่อตอบว่า input ดังกล่าวอยู่ class ใด
    # โดยจะตอบ class ที่ output node ของมันมีค่ามากที่สุด
    def predict_one(self, point):
        self.__forward(point)
        outputs = [node.result for node in self.outputLayer]
        max_index = outputs.index(max(outputs))
        return self.classes[max_index]
    
    # predict class ของข้อมูลหลายๆ ตัว
    def predict(self, data):
        return [self.predict_one(point) for point in data.rows]

    # ทำการสร้าง node และ link weight ตาม list ของ layers ที่ใส่ผู้ใช้กำหนดมา
    # โดย layer แรกของ list จะเป็น input layer และ layer สุดท้ายของ list จะเป็น output layer
    def setNetwork(self, layers_ls):
        is_input_layer = True
        prev_layer = []
        for layer in layers_ls:
            recent_layer = []
            for _ in range(layer.n):
                recent_layer.append(self.__addNeural(layer.activation, prev_layer))
            if is_input_layer:
                self.inputLayer = recent_layer
                is_input_layer = False
            else:
                self.hiddenLayer_ls.append(recent_layer)
            prev_layer = recent_layer
        self.outputLayer =  self.hiddenLayer_ls.pop()
    
    # เพิ่ม node ใน network โดยที่ node ดังกล่าวจะใช้ activation function ตามที่ให้มา
    # และจะมีเส้นเชื่อมเข้า node นี้จาก node ที่อยู่ใน precessors_ls
    def __addNeural(self, activation, precessors_ls):
        successor = Neural(activation)
        for precessor in precessors_ls:
            precessor.connect(successor)
        return successor
    
    # ทำการเปลี่ยนข้อมูลใน input layer ตาม input_ls ที่ให้มา
    # และคำนวณหา output ของแต่ละ node เรียงตามลำดับ layer จนถึง output layer
    def __forward(self, input_ls):
        for i in range(len(input_ls)):
            self.inputLayer[i].result = input_ls[i]
        for hiddenLayer in self.hiddenLayer_ls:
            for hidden in hiddenLayer:
                hidden.compute()
        for out in self.outputLayer:
            out.compute()
    
    # หาความต่างระหว่างค่าของ output ที่ควรจะเป็นกับ ค่าที่ได้จาก output node ในปัจจุบัน
    def __get_error(self, output_ls):
        error_ls = []
        for i in range(len(output_ls)):
            error_ls.append(output_ls[i] - self.outputLayer[i].result)
        return error_ls
    
    def __backprop(self, error_ls, lr):
        # reverse เพื่อคำนวณ local gradient ของแต่ละ node ใน network 
        # ไล่จาก node ใน output layer ถึง hidden layer แรกตามสมการที่ได้เรียนในชั้นเรียน
        self.hiddenLayer_ls.reverse()
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].updateGradOutput(error_ls[i])
        for hiddenLayer in self.hiddenLayer_ls:
            for hidden in hiddenLayer:
                hidden.updateGradHidden()
        
        # ปรับ weight โดยใช้ค่า local gradient ที่คำนวณไว้แล้วก่อนหน้านี้
        for out in self.outputLayer:
            out.updateWeightBias(lr)
        for hiddenLayer in self.hiddenLayer_ls:
            for hidden in hiddenLayer:
                hidden.updateWeightBias(lr)
        self.hiddenLayer_ls.reverse()

# Class ของ node ที่อยู่ใน NN
# จะเก็บข้อมูลเฉพาะของแต่ละ node ไว้รวมถึง edge ที่มีการเข้าออกจากมันด้วย
class Neural:
    def __init__(self, activation):
        self.activation = activation
        self.next = []
        self.prev = []
        self.bias = random.uniform(-1,1)
        self.localGrad = 0
        self.result = 0

    # ทำการสร้าง edge ระหว่าง node นี้ ที่ชี้ไปยัง node successor
    def connect(self, successor):
        edge = Edge(successor, random.uniform(-1,1), self)
        self.next.append(edge)
        successor.prev.append(edge)

    # คำนวณหา input ของ node (v) ตามสูตร sum(weight*x) + bias
    def __input(self):
        sum = 0
        for edge in self.prev:
            sum += edge.tail.result*edge.weight
        return sum + self.bias
    
    # คำนวณหา output ของ node (y) ตามสูตร f(v) เมื่อ f คือ activation function
    def compute(self):
        self.result = self.activation.compute(self.__input())
    
    # คำนวณหา local gradient ของ node ตามสูตรของ output node
    def updateGradOutput(self, error):
        self.localGrad = error*self.activation.diff(self.__input())

    # คำนวณหา local gradient ของ node ตามสูตรของ hidden node
    def updateGradHidden(self):
        sum = 0
        for edge in self.next:
            sum += edge.weight*edge.head.localGrad
        self.localGrad = self.activation.diff(self.__input())*sum

    # อัปเดต weight ทุก weight ที่เชื่อมเข้ามายัง node นี้
    # โดยใช้ local gradient ของ node นี้
    def updateWeightBias(self, lr):
        for edge in self.prev:
            edge.weight -= -lr*self.localGrad*edge.tail.result
        self.bias -= -lr*self.localGrad
        
# Class ที่ใช้เก็บข้อมูลของ edge แต่ละเส้น
class Edge:
    def __init__(self, head, weight, tail):
        self.weight = weight
        self.head = head
        self.tail = tail

# Class ที่ใช้เก็บข้อมูลของแต่ละ layer
class Layer:
    def __init__(self, n, activation):
        self.n = n
        self.activation = activation
