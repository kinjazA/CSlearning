# 第三章 01 布尔类型与比较运算符

## 01 布尔类型
> * 一般用于程序逻辑运算，进行判断，一般得出两种结果：是和否
> * 布尔类型的字面量：True（表示：真、是），False（表示：假、否）
> * **布尔类型可以和其他数据类型进行比较**，此时python会将**True视为1，False视为0**
> * **在python中，非0视为真值，0视为假值**

```python
b1 = False
b2 = True
print(b1 + 10)   # 结果为 10
print(b2 + 10)   # 结果为 11

if 0 :
    print("haha")
elif -1 :          # 判断条件写个字符串也行，也是非0
    print("ok")    # 代码结果为ok
```

## 02 比较运算符

> * 可以**通过比较运算符得到布尔类型**，即比较运算表达式的结果一定是True或False
> * 比较运算符的种类：`==`（判断是否相等），`！=`（判断是否不相等），`<`，`>`，`is`，`is not`等等
> * `is`判断两个变量引用对象是否为同一个，**根据对象在内存中的地址进行识别，和驻留机制有关**
> * `is not`判断两个变量引用对象是否不同
```python
# 定义变量存储布尔类型
bool_1 = True
bool_2 = False
print(f"bool_1变量的内容是{bool_1},数据类型为{type(bool_1)}")

# 通过比较运算符得出布尔类型
# ==, !=(一个感叹号一个等号), <, >, <=（一个小于号一个等号）, >= 六种，这里markdown显示与代码不一致，以实际代码为准
num1 = 10
num2 = 10
print(f"10==10的结果是{num1 == num2}")

a, b = 12, 42
flag = a > b
print(f"flage是{flag}")   # 结果为False
print(a is b)             # 结果为False
print(a is not b)         # 结果为True
```
## 03 二元布尔操作符

> * and和or操作符总是接收两个布尔值（或表达式），所以称之为二元                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
> * 若两个都为True，则and的结果为True；否则and的结果为False
> * 只要有一个为True，则or的结果为True；只有两个都是False，or的结果才为False

> 底层逻辑：
> * `x and y`：**若x为False，and会返回x的值；若x为True，and会返回y的值**
> * `x or y `：**若x为True，or会返回x的值；若x为False，or会返回y的值**
```python
a, b = 10, 4323
print(a and b)    # 结果为4323
print(a or b)     # 结果为10
print(not a)      # 结果为False，因为python中非0值视为True
```

## 04 not操作符

> * not操作符求值为**相反的布尔值**
> * not操作符还可以嵌套，就是类似于双重否定等于肯定，但比较少用
> * **在所有的布尔操作符中，python的运算顺序为先求not操作符，然后求and，最后求值or操作符**

```python
result = not not True
```



# 第三章 02 分支控制 if语句

## 01 基本格式

```
if 要判断的条件(为True):
	条件成立时，要做的事情   # 如果条件为False，则跳过子句
```
> * **判断语句的结果必须是布尔类型，True会执行if内的代码；False则不会执行**
> * python通过缩进（四个空格）来判断代码的归属关系
```python
age = int(input("请输入您的年龄："))
if age >= 18:
	print("您已成年，游玩需买票，10元")
print("祝您游玩愉快!")
```
## 02  if else 语句

> * 用于如果满足条件，做……，否则，做……
```python
print("欢迎来到游乐场，儿童免费，成人收费")
age = int(input("请输入您的年龄："))
if age >= 18:
	print(f"您的年龄为{age}，属于成年人，请付费游玩")
else：
	print(f"您的年龄为{age}，属于未成年人，可以免费游玩")
print("祝您玩得愉快！")
```
## 03 if elif else 语句

> * 运用于当出现多个判断条件时
> * **判断是有序的，满足上面的，下面的就不会执行，所以在多条件语句中，次序很重要**
>
```python
print("欢迎来到动物园！")
height = int(input("请输入你的身高（cm）："))
vip_level = int(input("请输入你的vip等级（1~5）："))
if height < 120:
    print("你的身高小于120cm，可以免费游玩")
elif vip_level > 3:
    print("您的vip等级大于3，可以免费游玩")
else:
    print("您需要付费游玩，票价为10元")
print("祝您玩得愉快！")

# 合并的写法
print("欢迎来到动物园！")
if int(input("请输入你的身高（cm）：")) < 120:
    print("你的身高小于120cm，可以免费游玩")
elif int(input("请输入你的vip等级（1~5）：")) > 3:
    print("您的vip等级大于3，可以免费游玩")
else:
    print("您需要付费游玩，票价为10元")
print("祝您玩得愉快！")
```

## 04 判断语句的嵌套
> * 适用于要进行二次或更多次的判断的多层判断需求、
> * 当外层条件满足时，执行内层的判断
> * if elif else可以自由组合
> * **最好不要超过三层，不然可读性很差**
```python
print("欢迎来到动物园！")
if int(input("请输入你的身高（cm）：")) > 120：
	print("你的身高超过120cm,需付费游玩")
	print("但若你的vip等级大于3，仍可以免费游玩")
	if int(input("请输入你的vip等级：")) > 3：
		print("可以免费游玩")
	else：
		print("需付费10元进行游玩")
else：
	 print("你可以免费游玩")
```
```python
# 例子：公司发礼物，需要满足两个条件，1.年龄大于18且小于30岁；2.入职时间大于两年或者级别大于3
if age > 18:
	if age < 30:
		if year > 2:
			print("可以领取礼物")
		elif level > 3:
			print("可以领取礼物")
		else：
			print("不符要求")
    else:
        print("年龄超过30，不符要求")
else:
	print("小于18岁，不符要求")
```

## 05 逻辑判断实例

```python
"""
定义一个数字（1~10），通过三次判断来猜出该数字
数字随机产生，每次没猜中，会提示大了还是小了，通过三层嵌套判断实现
"""
import random
num = random.randint(1, 10)
# 第一轮
guess_num = int(input("输入你猜测的数字："))
if guess_num == num:
    print("猜对了")
else:
    if guess_num > num:
        print("大了")
    else:
        print("小了")
        # 第二轮
    guess_num = int(input("第二次猜测的数字："))
    if guess_num == num:
        print("猜对了")
    else:
        if guess_num > num:
            print("大了")
        else:
            print("小了")
            # 第三轮
        guess_num = int(input("第三次猜测的数字："))
        if guess_num == num:
            print("猜对了")
        else:
            print("三次机会用完，猜错了")
```

