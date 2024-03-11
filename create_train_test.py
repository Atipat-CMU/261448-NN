import dotlis as dl
import random

# อ่านชุดข้อมูลทั้งหมดเข้ามา
table = dl.read_csv("datasets/cancer_preprocessed.txt")

test_percent = 35 # เปอร์เซ็นข้อมูลที่จะใช้ test เมื่อเทียบกับจำนวนข้อมูลทั้งหมด

# ทำการสร้างชุดข้อมูล train และ test ตามอัตราส่วน 65:35 ของข้อมูลทั้งหมด
# โดยแต่ละชุดข้อมูลจะยังคงอัตราส่วนของ class ไว้ตามข้อมูลต้นฉบับ
train = dl.DataTable(table.attributes, [])
test = dl.DataTable(table.attributes, [])

for c in table.unique("Class"):
    local_table = table.select(table.equal("Class", c))
    size = len(local_table.equal("Class", c))
    random_list = random.sample(local_table.rows, int(size * test_percent / 100))
    
    for _ in range(int(size * test_percent / 100)):
        random_sublist = random.choice(local_table.rows)
        local_table.rows.remove(random_sublist)
        test.rows.append(random_sublist)
    
    train.extend(local_table.rows)

random.shuffle(train.rows)
random.shuffle(test.rows)

# แปลงชุดข้อมูลทั้งสองเป็นไฟล์ csv เพื่อนำไปใช้ต่อในโปรแกรม main.py
train.to_csv("datasets/cancer_train.txt")
test.to_csv("datasets/cancer_test.txt")

# แสดงจำนวนข้อมูลที่ได้สำหรับชุดข้อมูล train และ test แยกตามแต่ละ class
print(f"Total number of rows in training set: {len(train.rows)}")
print(f"Number of rows with Class '2' in training set: {len(train.equal('Class', '2'))}")
print(f"Number of rows with Class '4' in training set: {len(train.equal('Class', '4'))}")
print()
print(f"Total number of rows in test set: {len(test.rows)}")
print(f"Number of rows with Class '2' in test set: {len(test.equal('Class', '2'))}")
print(f"Number of rows with Class '4' in test set: {len(test.equal('Class', '4'))}")
