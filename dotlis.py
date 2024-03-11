import random

# สร้างไว้เพื่อทำการจัดการข้อมูลในรูปแบบตาราง
# โดยได้รับแรงบันดาลใจมาจาก library pandas
class DataTable:
  def __init__(self, attributes, rows):
    self.attributes = attributes
    self.rows = rows
    self.num = len(rows) - 1

  # เปลี่ยน type ของข้อมูลในแต่ละ column ให้เป็น type ที่ต้องการ
  def retype(self, type_ls):
    for col_index, desired_type in enumerate(type_ls):
      column_values = self.column(col_index)
      converted_values = [desired_type(value) for value in column_values]
      for row_index, value in enumerate(converted_values):
        self.rows[row_index][col_index] = value

  # return list ของข้อมูลทั้งหมดใน column (index) นั้นๆ
  def column(self, number):
    return [row[number] for row in self.rows]
  
  # return list index ของข้อมูลเฉพาะแถวที่มีค่าตามที่ต้องการ
  def equal(self, attr_name, value_name):
    index = self.attributes.index(attr_name)
    values = self.column(index)
    return [i for i in range(len(values)) if values[i] == value_name]
  
  # return DataTable ที่ Drop column ตาม list ของ column index ที่ให้มา
  def drop(self, indices):
    return DataTable(self.attributes, self.__excludes(self.rows, indices))
  
  # return DataTable ที่ Drop column ตาม list ของชื่อ column ที่ให้มา
  def drop_column(self, list_of_column):
    tabel = DataTable(self.attributes, self.rows)
    if(is_type(list_of_column, str)):
      list_of_column = [tabel.attributes.index(column) for column in list_of_column]
    tabel.attributes = [self.attributes[i] for i in range(len(self.attributes)) if i not in list_of_column]
    tabel.rows = [[item for item in tabel.__excludes(row, list_of_column)] for row in tabel.rows]
    return tabel
  
  # return DataTable ที่เอามาเฉพาะ rows ที่ต้องการตาม list ของ row index ที่ให้มา
  def select(self, indices):
    return DataTable(self.attributes, self.__includes(self.rows, indices))
  
  # return set ของข้อมูลใน column ตามชื่อที่ให้มา
  def unique(self, attr_name):
    index = self.attributes.index(attr_name)
    return set(self.column(index))
  
  # เพิ่ม rows เข้าไปใน DataTable นี้
  def extend(self, rows):
    self.rows.extend(rows)

  # แยกตารางสำหรับการทำ Classification ได้ เพียงแค่ให้ชื่อ target feature ที่ต้องการให้ predict
  def exclass(self, attr_name):
    index = self.attributes.index(attr_name)
    y = DataTable(self.attributes[index], [[row[index]] for row in self.rows])
    X = DataTable(self.__excludes(self.attributes, [index]), self.drop_column([index]).rows)
    return X, y
  
  # return สมาชิกที่ไม่ได้มี index อยู่ใน list ของ index ที่ให้มา
  def __excludes(self, list, indices):
    return [list[i] for i in range(len(list)) if i not in indices]
  
  # return สมาชิกที่มี index อยู่ใน list ของ index ที่ให้มา
  def __includes(self, list, indices):
    return [list[i] for i in range(len(list)) if i in indices]
  
  # สร้าง file .txt จาก DataTable ที่แบ่งข้อมูลโดยใช้ ,
  def to_csv(self, filename):
    with open(filename, "w") as f:
      f.write(','.join(self.attributes) + '\n')
      for row in self.rows:
        f.write(','.join(map(str, row)) + '\n')
  
# สร้าง DataTable จาก file .txt ที่แบ่งข้อมูลโดยใช้ tab
def read_tsv(filename):
  f = open(filename, "r")
  source = f.read()
  rows = list(filter(lambda item: item != "", source.split("\n")))

  rows = [row.split("\t") for row in rows]
  attributes = rows[0]
    
  return DataTable(attributes, rows[1:])

# สร้าง DataTable จาก file .txt ที่แบ่งข้อมูลโดยใช้ ,
def read_csv(filename):
  f = open(filename, "r")
  source = f.read()
  rows = list(filter(lambda item: item != "", source.split("\n")))

  rows = [row.split(",") for row in rows]
  attributes = rows[0]
    
  return DataTable(attributes, rows[1:])

# สร้าง DataTable train และ test โดยการสุ่มเลือกตามสัดส่วนเปอร์เซ็น
# โดยแต่ละ class จะถูกเลือกในอัตราส่วนที่เท่ากัน
def slipt_test_by_class(table, attr_name, percent):
  train = DataTable(table.attributes, [])
  test = DataTable(table.attributes, [])
  for c in table.unique(attr_name):
    local_table = table.select(table.equal(attr_name, c))
    size = len(local_table.rows)
    random_list = random.sample(local_table.rows, int(size*percent/100))
    for _ in range(int(size*percent/100)):
      random_sublist = random.choice(local_table.rows)
      local_table.rows.remove(random_sublist)
      test.rows.append(random_sublist)
    train.extend(local_table.rows)
  return train, test

# return boolean ใช้ตรวจสอบว่าสมาชิกทุกตัวของ list มี type ตามที่ต้องการหรือไม่
def is_type(list, desired_type):
    return all(isinstance(element, desired_type) for element in list)
