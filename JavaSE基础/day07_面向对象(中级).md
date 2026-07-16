# 01 IDEA的使用

> * 源码写在src文件夹中
> * 编译后的`.class`文件在out文件夹中

## 01.1 常用快捷键

> * 在`setting`里的`Keymap`里可以查看
> * 删除行`ctrl + Y`
> * 补全代码`alt + /`
> * 快速格式化代码`ctrl + alt +L`
> * ==生成构造器`alt + insert`==
> * 查看类的继承关系`ctrl + H`
> * 快速定位方法的代码位置`ctrl + B`
> * 自动分配变量名 写完方法后加`.var`，再回车

```java
//演示
new 类名().var  //再回车
```

## 01.2 IDEA模板

> * 在setting里的Editor的Live Template里可以查看，可以自定义
> * 输入之后回车即可生成对应的代码语句
> * 生成一个主方法`main`，再回车
> * 生成一个for循环`fori`，再回车
> * 生成输出语句`sout`，再回车

# 02 包 package

## 02.1 包的结构

> * 包是一种用于组织和管理类的方式，它提供了一种将类按功能和用途分组的方法。包的概念在Java中非常重要，它有助于维护大型项目的可读性和可管理性
>
> * 包提供了一种命名空间的机制，==可以避免类的名称冲突。不同包中的类可以使用相同的名称，因为它们在不同的命名空间中==
>
> * 包提供了一种封装的机制，允许开发者隐藏内部实现细节，只暴露必要的接口
>
> * 包的名称通常使用小写字母，并且==采用反向域名的命名方式==，如` com.examplecorp.myapp.module`，一个点就是一层文件夹
>

```java
src/
  └── com/
        └── examplecorp/
                    └── myapp/
                          └── module/
```

> * 在IDEA中的src文件夹中，创建package，命名方式采用反向域名，然后每个包里可以创建java类文件，不同包的类名可以相同

![](day07_面向对象(中级).assets/image-20240721082440969.png)

## 02.2 不同包里的同名类调用
> * 在一个类里调用两个同名类的情况：
> * ==先输入`new 类名`会弹出选择是哪个文件夹下的类，然后尾部输入`.var`自动生成变量==
> * 第二个同名类调用时同样的步骤，但是代码里会显示出他的包名

![](day07_面向对象(中级).assets/image-20240721132332623.png)

## 02.3 包的命名 
> 包的命名规则：
>
> * 只能包含数字、字母、下划线、小圆点，不能以数字开头，不能是关键字或是保留字
> * 一般是小写字母+小圆点

```java
//一般写法
com.公司名.项目名.业务模块名
com.sina.crm.user //用户模块
com.sina.crm.order  //用户模块
com.sina.crm.utils  //工具类
```

## 02.4 包的导入

> * 导入（Import）是将其他包中的类或整个包中的所有类引入到当前文件中，以便在当前文件中使用这些类而不需要每次都指定完整的类名
> * ==在同一个包里的不同类可以直接使用，不用导入，new个对象就可以用了==

> * 导入单个类

```java
//基本语法
import 包路径.类名;

import java.util.ArrayList;
```

> * 导入整个包
> * 需要使用某个包中的多个类，可以导入整个包，这样你就不需要为每个类单独编写导入语句

```java
//基本语法
import 包路径.*;

import java.util.*;
```

> * 静态导入
> * Java还允许静态导入，这允许你导入特定的静态方法或静态字段，而不需要每次都使用类名作为前缀

```java
//基本语法
import 包路径.类名.静态成员;
//或者
import static 包路径.类名.静态成员;

import static java.lang.Math.PI;
import static java.lang.Math.pow;
```

> * 选择性导入允许从一个包中导入多个特定的类，但不是整个包

```java
//基本语法
import 包路径.{类名1, 类名2};

import java.util.{ArrayList, HashMap};
```

> * 在Java源文件中，导入语句应该放在包声明之后，任何其他代码（如类定义、方法定义）之前。如果有多个导入语句，通常按照以下顺序排列：
>   1. 所有`java.*`和`javax.*`的导入
>   2. 第三方库的导入
>   3. 当前项目的包导入

## 02.5 包的使用

```java
import java.util.Arrays;

public class Import01 {
    public static void main(String[] args) {
        int[] arr = {-4231, 632, 73654, 847};
        //调用Arrays的sort方法实现排序
        Arrays.sort(arr);
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
    }
}
```

> * package的作用是声明当前类所在的包，需要放在类的最上面，一个类中至多一句package
> * import语句放在package的下面，在类定义前面

```java
package com.xiaoming;

import java.util.Scanner;

public class Dog{
    public static void main(String[] args) {
        Scanner sc1 = new Scanner(System.in);
        System.out.println("请输入一个整数");
        int a = sc1.nextInt();
        System.out.println("你输入的整数是：" + a);
    }
}
```

# 03 访问修饰符

## 03.1 快速入门

> * java一共有四种访问控制修饰符，用于控制方法和属性（成员变量）的访问权限（范围）
> * 公开级别：`public`，对外公开
> * 受保护级别：`protected`，对子类和同一个包中的类公开
> * 默认级别：没有修饰符号，向同一个包的类公开，子类不行
> * 私有级别：`private`，只有类本身可以访问，不对外公开

![](day07_面向对象(中级).assets/838bd85fe591ebae4d1ce59581eeaf1.jpg)


```java
package com.xiaoming;

public class ParentClass {
    //属性
    public int publicVar = 10; // 可以被任何类访问
    protected int protectedVar = 20; // 可以被同一包中的类和子类访问
    int packagePrivateVar = 30; // 只能被同一包中的类访问
    private int privateVar = 40; // 只能在这个类中访问
	
    //方法
    public void publicMethod() {} // 可以被任何类访问
    protected void protectedMethod() {} // 可以被同一包中的类和子类访问
    void packagePrivateMethod() {} // 只能被同一包中的类访问
    private void privateMethod() {} // 只能在这个类中访问
}
```

# 04 面向对象三大特征 *

## 04.1 封装 encapsulation

> * 就是把抽象出来的属性和方法封装在一起，数据被保护在内部，程序的其他部分只有通过被授权的方法才能对数据进行操作。比如说用遥控器操作电视，其实就相当于电视机的功能封装在遥控器中
> * 可以隐藏实现的细节、可以对数据进行验证，保证安全合理

### 04.1.1 封装实现步骤

> * ### 实现封装的步骤：
>
>   1. **定义类**： 创建一个类，该类将包含封装的属性和方法。
>   2. ==**私有化属性**： 将类的属性（成员变量）声明为`private`，这样它们就不能被外部类直接访问(注意类内方法不要泄露了)==
>   3. **提供公共方法**： 提供公共的getter和setter方法，允许外部类访问和修改私有属性。这些方法允许控制对属性的访问，确保数据的完整性
>   4. **实现构造方法**： 提供构造方法来初始化对象的状态。
>   5. **实现其他方法**： 实现其他公共方法，这些方法可以执行更复杂的操作，同时访问和修改私有属性。


```java
// 定义一个名为Person的类
public class Person {
    // 私有化属性，外部类不能直接访问这些属性
    private String name;
    private int age;

    // 提供公共的getter方法，允许外部类访问私有属性
    public String getName() {
        return this.name; // 返回name属性的值
    }

    // 提供公共的setter方法，允许外部类修改私有属性
    public void setName(String name) {
        this.name = name; // 重置name属性的值
    }

    // 提供公共的getter方法，允许外部类访问私有属性
    public int getAge() {
        return this.age; // 返回age属性的值
    }

    // 提供公共的setter方法，允许外部类修改私有属性
    public void setAge(int age) {
        if (age > 0) { // 检查年龄是否合理
            this.age = age; // 设置age属性的值
        }
    }

    // 一个公共方法，展示个人信息
    public void displayInfo() {
        System.out.println("Name: " + name + ", Age: " + age);
    }
}

// 测试Person类的主类
public class Main {
    public static void main(String[] args) {
        // 创建Person对象
        Person person = new Person("John Doe", 30);

        // 使用getter方法获取属性值
        System.out.println("Initial Name: " + person.getName());
        System.out.println("Initial Age: " + person.getAge());

        // 使用setter方法修改属性值
        person.setName("Jane Doe");
        person.setAge(25);

        // 再次使用getter方法获取并打印属性值
        System.out.println("Updated Name: " + person.getName());
        System.out.println("Updated Age: " + person.getAge());

        // 调用公共方法展示个人信息
        person.displayInfo();
    }
}
```

==注意上面的代码是两个public类，不能在同一个java文件中==

> * 如果属性过多，一个个写get和set方法太麻烦，可以使用快捷键`alt + insert`，选择getter and setter自动创建  

### 04.1.2 构造器与setXXX组合

> * 在Java中，构造器（constructor）和setter方法（通常称为setters）都是用来初始化对象的属性，但是它们的使用场景和目的有所不同。构造器在对象创建时被调用，而setter方法可以在对象创建之后的任何时间点调用，用于修改对象的状态。
> * ==为了保持封装性，我们通常将类的属性设为private，然后通过公共的setter方法来修改这些属性。这样做的好处是，可以在setter方法中添加额外的逻辑，比如参数检查、数据转换等，而不仅仅是简单地赋值。==
> * 然而，在构造器中直接使用属性赋值，可能会绕过setter方法中的这种额外逻辑。为了避免这种情况，可以采用以下几种策略
> * ==方法一==：**在构造器中调用setter方法： 在构造器内部，可以显式地调用setter方法来设置属性的值。这样可以确保所有的属性设置都经过了setter方法的处理，包括在对象创建时的初始化**

```java
class Person{
    private String name;
    private int age;
    private double salary;
    //构造器
    public Person(String name, int age, double salary) {
        //先是属性的初始化
        this.name = name;
        this.age = age;
        this.salary = salary;
        //再调用setters方法确保诸如数据验证或其他逻辑不漏掉
        setAge(age);   //也可以 this.setAge(age);就结合了上面的，不用输这么多
        setSalary(salary);
        setName(name);
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        if (age > 0 && age <= 120) { // 检查年龄是否合理
            this.age = age; // 设置age属性的值
        }else{
            System.out.println("年龄不合法");
            this.age = -1;  //设一个默认值
        }
    }

    public double getSalary() {
        return salary;
    }

    public void setSalary(double salary) {
        this.salary = salary;
    }
}

public class Main {
    public static void main(String[] args) {
        Person person = new Person("张三", 2666, 5000);
        System.out.println(person.getName());
        System.out.println(person.getAge());
        System.out.println(person.getSalary());
    }
}
```

### 04.1.3 练习

```java
//创建两个类，一个AccountTest，一个Account，Account类有姓名、余额、密码三个属性

//AccountTest文件
package com.encap;

public class AccountTest {
    public static void main(String[] args) {
        Account acc1 = new Account(20, "张三", "12346");
        System.out.println(acc1.getBalance());
        System.out.println(acc1.getName());
        System.out.println(acc1.getKeywords());
    }
}

//Account文件
package com.encap;

class Account {
    private double balance;
    private String name;
    private String keywords;

    public Account(double balance, String name, String keywords){
        this.setBalance(balance);
        this.setName(name);
        this.setKeywords(keywords);
    }

    public double getBalance() {
        return this.balance;
    }

    public void setBalance(double balance) {
        if (balance >= 20) {
            this.balance = balance;
        } else {
            System.out.println("余额不足");
        }
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        if (name.length() >= 2 && name.length() <= 4) {
            this.name = name;
        } else {
            System.out.println("名字长度不符合要求");
            this.name = "无名";
        }
    }

    public String getKeywords() {
        return keywords;
    }

    public void setKeywords(String keywords) {
        if(keywords.length() == 6){
            this.keywords = keywords;
        }else{
            System.out.println("密码长度不符合要求");
            this.keywords = "000000";
        }
    }
}
```

## 04.2 继承 Inheritance

### 04.2.1 继承原理

> * 继承是一种强大的机制，允许创建新的类，这些类可以从现有的类那里继承属性和行为。这种关系被称为“is-a”关系，意味着一个类（子类或派生类）是另一个类（父类或基类）的一种类型，即子类应该是父类的一个特殊化
> * 可以提高代码的复用，简化代码

```java
//基本语法
class 子类 extends 父类{
    
}
```

> * 子类会自动拥有父类定义的属性和方法
> * 父类又叫做超类、基类
> * 子类又叫做派生类

![](day07_面向对象(中级).assets/362ca2cee500cd7d29ab97759d3a853.jpg)

### 04.2.2 继承的示例

> * Student类为父类，Pupil和Graduate都是子类，extend_为调用文件
> * 代码结构如下

![](day07_面向对象(中级).assets/image-20240724091450086.png)

```java
//Student类文件
package com.extend01;
//用作pupil和graduate的父类
public class Student {
    //共有的属性
    public String name;
    public int age;
    private double score;
    //共有的方法
    public void setScore(double score) {
        this.score = score;
    }

    public void showInfo() {
        System.out.println("学生信息：" + name + " " + age + " " + score);
    }
}
```

```java
//Pupil类文件
package com.extend01;

public class Pupil extends Student{
    public void testing() {
        System.out.println("小学生" + name + "正在考小学数学");
    }
}
```

```java
//Graduate类文件
package com.extend01;

public class Graduate extends Student{
    public void testing() {
        System.out.println("大学生" + name +"正在考大学数学");
    }
}
```

```java
//Extend_文件
package com.extend01;

public class Extend_ {
    public static void main(String[] args) {
        Pupil pupil = new Pupil();
        pupil.age = 10;
        pupil.name = "小明";
        pupil.setScore(60.5);
        pupil.showInfo();
        pupil.testing();
        System.out.println("-------------------");

        Graduate graduate = new Graduate();
        graduate.age = 20;
        graduate.name = "小红";
        graduate.setScore(80.5);
        graduate.showInfo();
        graduate.testing();
    }
}
```

> * 其实就是一种==抽象出共性的思想==，从每个具体的情况中抽象出共性，把共性的写成父类，具体 情况就都能继承这个父类，并且根据具体来赋予不同的比方说属性信息

### 04.2.3 继承的细节

> * ==子类继承了所有属性和方法，但是私有属性和方法不能在子类直接访问，要通过公共的方法（父类的）去访问==

![](day07_面向对象(中级).assets/image-20240724105537528.png)

```java
//Base类文件
package com.extend01.ExtendDetail;

public class Base {  //基类
    public int n1 = 100;
    protected int n2 = 200;
    int n3 = 300;
    private int n4 = 400;

    public Base() {
        System.out.println("Base()...");  //构造方法，会被自动调用
    }

    public void test100(){
        System.out.println("test100");
    }
    protected void test200(){
        System.out.println("test200");
    }
    void test300(){
        System.out.println("test300");
    }
    private void test400(){
        System.out.println("test400");
    }
    //提供公共方法，用以子类间接访问private属性
    public int Getn4(){
        return n4;
    }
    //提供公共方法调用私有方法
    public void Calltest400(){
        test400();
    }
}
```

```java
//Sub类文件
package com.extend01.ExtendDetail;

public class Sub extends Base{

    public Sub() {
        System.out.println("Sub()...");  //会被自动调用
    }

    public void Sayok(){
        System.out.println("n1=" + n1 + "n2=" + n2 + "n3=" + n3);
        //System.out.println("n4=" + n4);这句不行，n4是private，不能直接调用
        int n4_ = Getn4();
        System.out.println("n4_=" + n4_);
        test100();
        test200();
        test300();
        //test400();这句不行，test400()是private，不能直接调用
        Calltest400();
    }
}
```

```java
//Test类文件
package com.extend01.ExtendDetail;

public class Test {
    public static void main(String[] args) {
        Sub sub = new Sub();
        sub.Sayok();
    }
}
```

> * ==子类必须调用父类的构造器，完成父类的初始化==                                                

```java
class Parent {
    Parent() {  //无参构造器
        System.out.println("Parent Constructor Called");
    }
}

class Child extends Parent {
    Child() {  //无参构造器
        // 隐式调用super(); 即调用父类的无参构造器
        System.out.println("Child Constructor Called");
    }
}

public class Test {
    public static void main(String[] args) {
        Child child = new Child();
        // 输出: Parent Constructor Called Child Constructor Called
    }
}
```

> * 创建子类时，不管使用子类的哪个构造器，==默认情况下总会自动调用父类的无参构造器，如果父类没有提供无参构造器，则必须在子类的每个构造器中使用`super`去指定使用父类的哪个构造器完成父类的初始化==
> * 在创建子类对象时，会默认调用父类的无参构造器来初始化继承的成员变量。==子类可以通过`super`关键字显式调用父类的其他构造器(必须写在子类构造器第一句)==。这个机制确保了父类的正确初始化      

```java
class Parent {
    int value;

    // 父类只有一个有参构造器，没有无参构造器
    Parent(int value) {
        this.value = value;
        System.out.println("父类有参构造器被调用，参数为：" + value);
    }
}

class Child extends Parent {
    // 子类构造器必须显式调用父类的有参构造器
    Child() {
        super(10); // 显式调用父类的有参构造器，并传入参数10
        System.out.println("子类无参构造器被调用");
    }

    Child(int childValue) {
        super(childValue); // 显式调用父类的有参构造器，并传入参数childValue
        System.out.println("子类有参构造器被调用，参数为：" + value);
    }
}

public class Test {
    public static void main(String[] args) {
        new Child(); //输出: 父类有参构造器被调用，参数为：10     子类无参构造器被调用

        new Child(20); //输出:父类有参构造器被调用，参数为：20  子类有参构造器被调用，参数为：20
    }
}
```

我自己的理解是子类一定要执行父类的一个构造器，`super（）`语句相当于就是在指定某个父类的构造器，应该是根据括号内的参数类型来判断到底是哪个构造器，从而把参数传入父类的构造器。第二个对象走的是子类的有参构造器，这个构造器的参数写的和super的参数一致，从而父类和子类的参数保持一致

> * `super`只能在构造器中使用
> * ==`super`和`this`(这里指this用于在一个构造器里调用另一个构造器的情况)都要放在构造器里的第一句，所以这两个不可以出现在同一个构造器==(这里还存疑)
> * ==java中所有类都是`Object`类的子类，IDEA中鼠标定位到某个类，输入`ctrl + H`可以查看类的继承关系==

![](day07_面向对象(中级).assets/image-20240724195725637.png)

> * ==父类构造器的调用不限于直接父类，将一直往上追溯，直到Object类(顶级父类)。==也就是不仅爸爸的属性和方法可以用，爷爷的也可以，曾爷爷......一直往上到猴子都可以用。==构造器则是从猴子往下一层层调用直到子类的构造器==

> * ==子类最多继承一个父类（指直接继承）==，即java是单继承机制，同时，最好不要滥用继承，要满足`子类 is a 父类`这样的关系才考虑用继承

### 04.2.4 继承的本质

> * 下面涉及继承的代码执行时的内存情况

```java
public class Test{
    public static void main(String[] args) {
        Son son1 = new Son();
    }
}

class Grandpa{
    String name = "大头爷爷";
    String hobby = "旅游";
}
class Father extends Grandpa{
    String name = "大头爸爸";
    int age = 35;
}
class Son extends Father{  //子类
    String name = "大头儿子";
}
```

==当创建对象`son1`时，会先加载`Object`类信息，再加载`Grandpa`类信息，再加载`Father`类信息，再加载`Son`类信息==

![](day07_面向对象(中级).assets/60fce49a4b00246a01511aea8495cfb.jpg)

在这个例子中，三个类都有`name`这个属性，如果我们用`对象.name`去访问，会是什么结果呢？在java中，会根据查找关系来返回信息

> * 在继承中，首先会查找子类是否有该属性或是方法，如果有且可以成功访问，则返回信息
> * ==比如说不在同一个类，属性修饰符为`private`，即不能成功访问，则代码会报错==
> * 如果没有或是不能成功访问，则向上查找父类中是否有，如果有且可以成功访问，则返回信息
> * 如果父类没有，则继续向上，直到整个继承链查找完毕

```java
public class Test{
    public static void main(String[] args) {
        Son son1 = new Son();
        System.out.println(son.name);  //报错
    }
}

class Grandpa{
    String name = "大头爷爷";
    String hobby = "旅游";
}
class Father extends Grandpa{
    String name = "大头爸爸";
    int age = 35;
}
class Son extends Father{  //子类
     private String name = "大头儿子";  //这里访问修饰符进行限制
}
```

![](day07_面向对象(中级).assets/image-20240724224106224.png)

> * 这里可以结合上面提到的`get`方法间接访问到该属性

```java
public class Test{
    public static void main(String[] args) {
        Son son1 = new Son();
        System.out.println(son1.Getname());  //间接访问
    }
}

class Grandpa{
    String name = "大头爷爷";
    String hobby = "旅游";
}
class Father extends Grandpa{
    String name = "大头爸爸";
    int age = 35;
}
class Son extends Father{  //子类
     private String name = "大头儿子";

     public String Getname(){  //获取属性
         return name;
     }
}
```

### 04.2.5 练习

> * 分析下面代码的输出结果

```java
class A{
    A(){
        System.out.println("a");
    }
    A(String name){
        System.out.println("a name");
    }
}
class B extends A{
    B(){
        this("abc");     //这里不能有，因为已经有this
        System.out.println("b");
    }
    B(String name){
        System.out.println("b name");  //原因在于其实隐含了这句前有个默认的super()
    }
}
public class  Test{
    public static void main(String[] args){
        B b = new B();
    }
}
```

结果为：

a

 b name

 b

> * 判断下面代码的输出结果

```java
class A{
    public A(){
        System.out.println("我是A类");
    }
}
class B extends A{
    public B(){
        System.out.println("我是B类的无参构造器");
    }
    public B(String name){
        System.out.println("我是B类的有参构造器，参数为："+name);
    }
}
class C extends B{
    public C(){
        this("hello");
        System.out.println("我是C类的无参构造器");
    }
    public C(String name){
        super("hahaha");
        System.out.println("我是C类的有参构造器，参数为："+name);
    }
}

public class Test{
    public static void main(String[] args)
    {
        C c = new C();
    }
}
```

结果为 ：

我是A类
我是B类的有参构造器，参数为：hahaha
我是C类的有参构造器，参数为：hello
我是C类的无参构造器

> * 创建Computer类、PC类、NotePad类用以展示信息

```java
package com.extend01.ExtendDetail;

public class Computer {
    public String cpu;
    public String ram;
    public String hdd;
    Computer(String cpu, String ram, String hdd){  //构造器
        this.cpu = cpu;
        this.ram = ram;
        this.hdd = hdd;
    }
    public void show(){
        System.out.println("CPU:"+cpu+" RAM:"+ram+" HDD:"+hdd);
    }
}
```

```java
package com.extend01.ExtendDetail;

public class PC extends Computer{
    public String brand;
    public PC(String cpu, String ram, String hdd, String brand) {
        super(cpu, ram, hdd);  //这里只使用了super来调用父类Computer的构造器，而没有使用this
        this.brand = brand;  //这一行是再设置子类特有的brand属性，这并不违反规则，因为它并没有调用另一个构造器
    }
    public void show() {
        super.show();
        System.out.println("品牌：" + brand);
    }
}
```

```java
package com.extend01.ExtendDetail;

public class NotePad extends Computer {
    public String color;
    NotePad(String cpu, String ram, String hdd, String color){
        super(cpu,ram,hdd);
        this.color = color;
    }
    public void show(){
        super.show();
        System.out.println("颜色："+color);
    }
}
```

```java
package com.extend01.ExtendDetail;

public class Test {
    public static void main(String[] args){
        PC pc = new PC("i5", "8G", "1T", "Windows");
        pc.show();
        System.out.println("=======================");
        NotePad np = new NotePad("i7", "16G", "2T", "灰色");
        np.show();
    }
}
```

## 04.3 super关键字

> * ==`super` 关键字在Java中有两个主要用途：访问父类的属性和方法，以及调用父类的构造器==
> * 个人理解其实就是父类的一个代名词
> * ==当子类需要访问父类中定义的属性或方法时(private的除外)，可以使用`super`关键字==。这在子类需要扩展或修改父类行为时非常有用。使用`super`可以避免直接访问父类的成员造成的混淆，尤其是在子类有与父类同名成员的情况
> * ==当父类和子类有重名属性或是方法时，必须使用`super`来进行访问。若没有重名，`super`、`this`和直接访问效果是一样的，感觉还是用`super`比较不容易出错==

```java
class Parent {  
    int number = 10;  
    void show() {  
        System.out.println("Parent show()");  
    }  
}  

class Child extends Parent {  
    int number = 20;  
    void show() {  
        System.out.println("Child show()");  
        System.out.println("Child number = " + number);  
        System.out.println("Parent number = " + super.number); // 使用super访问父类的number  
        super.show(); // 使用super调用父类的show()方法  
    }  
}  

public class Test {  
    public static void main(String[] args) {  
        Child child = new Child();  
        child.show();  
    }  
}
```

> * 在子类的构造器中，`super`关键字也可以用来指代父类的构造器。这是初始化父类部分的必要步骤，因为子类继承自父类，所以需要先构造父类部分。==如果父类没有无参构造器，那么子类必须通过`super`明确指定调用父类的哪个构造器，且必须放在构造器里的第一句==
> * ==不局限于父类，如果爷爷类的属性或是方法要访问，也可以用，super会遵循就近原则：也就是先找直接父类，没有的话，再往上一级父类找==

```java
class Parent {  
    int number;  
  
    Parent(int number) {  
        this.number = number;  
        System.out.println("Parent Constructor with number: " + number);  
    }  
  
    Parent() {  
        System.out.println("Parent No-arg Constructor");  
    }  
}  
  
class Child extends Parent {  
    Child() {  
        // 调用父类的有参构造器  
        super(10);  
        System.out.println("Child Constructor");  
    }  
}  
  
public class Test {  
    public static void main(String[] args) {  
        Child child = new Child();  
    }  
}
```

![](day07_面向对象(中级).assets/e496e129297220be629ca5897e71293.jpg)

## 04.4 方法重写/覆盖

> * 在Java中，方法重写（Override）是面向对象编程中的一个重要概念，它==允许子类提供一个特定的实现，用于替换父类中某个方法的实现==。这是实现多态性的关键机制之一
>
>   ### 方法重写的规则
>
>   1. **方法签名必须相同**：==子类重写的方法必须与父类中被重写的方法具有相同的方法名称、参数列表（包括参数的类型和顺序）==
>
>   2. **返回类型兼容**：==子类重写的方法的返回类型必须与父类中被重写的方法的返回类型相同，或者是其子类型（称为协变返回类型）==
>
>   3. **访问权限不能更严格**：子类重写的方法不能拥有比父类中被重写的方法更严格的访问级别。例如，如果父类方法是`protected`，那么子类重写的方法不能是`private`
>
>   4. **抛出的异常不能更广泛**：子类重写的方法抛出的异常类型应该是父类方法抛出异常类型的子集或相同
>
>   5. **实例方法不能重写静态方法**：如果父类中的方法是静态的，那么子类不能通过重写将其变为实例方法（反之亦然）
>
>   6. **不是所有的方法都可以被重写**：例如，`final`方法、`static`方法（如果子类没有将其变为实例方法）、`private`方法等都不能被重写
>
>   7. **使用@Override注解（可选）**：在子类中重写父类方法时，可以使用`@Override`注解来显式声明该方法是一个重写方法。这不是必需的，但它有助于编译器检查方法签名是否正确，从而提高代码的可读性和可维护性
>
>  ### 方法重写的目的
>
>  - **实现多态性**：通过方法重写，子类可以提供与父类相同方法签名的具体实现，从而实现运行时多态
>   - **增强功能**：子类可以通过重写父类的方法来添加或修改功能，以满足特定的需求
>   - **提高代码的可维护性和可扩展性**：通过方法重写，可以在不修改父类代码的情况下，通过继承和多态来扩展和修改类的行为

```java
class Animal {  
    void eat() {  
        System.out.println("This animal eats food.");  
    }  
}  
  
class Dog extends Animal {  
    @Override  //可选
    void eat() {  
        System.out.println("Dog eats meat.");  
    }  
}  
  
public class TestOverride {  
    public static void main(String[] args) {  
        Animal myDog = new Dog();  
        myDog.eat(); // 输出 "Dog eats meat."  
    }  
}
```

## 04.5 重写VS重载

![](day07_面向对象(中级).assets/27d4a8cb1b98d1feee84ddf0dff4871.jpg)

```java
class Person{
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void Say() {
        System.out.println("hello,my name is " + name);
    }

    public String getName(){
        return name;
    }
}

class Student extends Person{
    private String id;
    private int score;

    public Student(String name ,int age,String id,int score){
        super(name, age);
        this.id = id;
        this.score = score;
    }
    public void Say(){
        System.out.println("hello,my name is " + super.getName() + ",and my id is " + id + ",and my score is " + score);
    }
}

public class Main{
    public static void main(String[] args) {
        Student s = new Student("zhangsan", 18, "2019001", 100);
        s.Say();
    }
}
```

## 04.6 多态 Polymorphism * 难

> * 多态是面向对象编程中的一个重要概念，==它允许对象以多种形式出现。多态性通过方法的重载和重写实现，使得相同的操作可以应用于不同的对象上，并且在执行时表现出不同的行为==
> * ==通过使用父类引用指向子类对象，可以实现多态性，从而使代码更加灵活、可扩展、易于维护。直接使用子类对象则会失去这些优势，导致代码重复、耦合性高，维护和扩展变得困难==
> * 一个对象的编译类型和运行类型可以不一致，编译类型在定义对象时就确定了，不能改变，运行类型是可以变化的
> * ==编译类型看定义时等号的左边，运行类型看等号的右边==

```java
Animal animal = new Dog();  //animal编译类型时Animal，运行类型时Dog
animal = new Cat();  //这里animal的运行类型就变成了Cat，而它的编译类型仍是Animal
```

### 04.6.1 编译时多态（方法重载）

> * 编译时多态是在编译阶段决定调用哪个方法。方法重载是实现编译时多态的一种方式。同一个类中可以有多个方法名相同但参数不同的方法

```java
class MathOperation {
    // 方法重载示例：两个add方法，参数类型不同
    public int add(int a, int b) {
        return a + b;
    }

    public double add(double a, double b) {
        return a + b;
    }
}

public class Main {
    public static void main(String[] args) {
        MathOperation operation = new MathOperation();
        System.out.println(operation.add(2, 3));          // 输出：5
        System.out.println(operation.add(2.5, 3.5));      // 输出：6.0
    }
}
```

### 04.6.2 运行时多态（方法重写）

> * 运行时多态是在运行时决定调用哪个方法。方法重写是实现运行时多态的一种方式。==子类可以重写父类的方法，使得在运行时通过父类引用调用子类的重写方法==

```java
class Animal {
    // 父类的方法
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }
}

class Dog extends Animal {
    // 重写父类的方法
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }
}

class Cat extends Animal {
    // 重写父类的方法
    @Override
    public void makeSound() {
        System.out.println("Meow");
    }
}

public class Main {
    public static void main(String[] args) {
        // 父类引用指向子类对象
        Animal myDog = new Dog();
        Animal myCat = new Cat();

        myDog.makeSound(); // 输出：Woof，执行到该行时，myDog的运行类型是Dog，故用其对应方法
        myCat.makeSound(); // 输出：Meow
    }
}
```

```java
//示例一：统一处理不同类型的子类对象
class Animal {
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow");
    }
}

public class Main {
    public static void main(String[] args) {
        // 使用父类引用指向子类对象
        Animal myDog = new Dog();
        Animal myCat = new Cat();

        // 统一处理不同类型的子类对象
        makeAnimalSound(myDog);
        makeAnimalSound(myCat);
    }

    public static void makeAnimalSound(Animal animal) {
        animal.makeSound(); // 动态绑定，根据实际对象类型调用相应的方法
    }
}
```

```java
//示例二：使用集合存储和处理不同类型的子类对象
import java.util.ArrayList;
import java.util.List;

class Animal {
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow");
    }
}

public class Main {
    public static void main(String[] args) {
        // 创建一个父类类型的List来存储子类对象
        List<Animal> animals = new ArrayList<>();
        animals.add(new Dog());
        animals.add(new Cat());

        for (Animal animal : animals) {
            animal.makeSound(); // 动态绑定，根据实际对象类型调用相应的方法
        }
    }
}
```

```java
//直接使用子类对象的问题
public class Main {
    public static void main(String[] args) {
        Dog myDog = new Dog();
        Cat myCat = new Cat();

        // 需要为每一种子类对象编写不同的方法
        makeDogSound(myDog);
        makeCatSound(myCat);
    }

    public static void makeDogSound(Dog dog) {
        dog.makeSound(); // 只能处理Dog类型的对象
    }

    public static void makeCatSound(Cat cat) {
        cat.makeSound(); // 只能处理Cat类型的对象
    }
}
```

### 04.6.3 多态的细节 **

> * 多态的前提：两个对象（类）存在继承关系
> * 可以点击左侧的sturcture来查看类的method和field

![](day07_面向对象(中级).assets/image-20240729222243613.png)

> * ==在使用父类引用指向子类对象时，可以访问父类中定义的成员（方法和属性），但不能直接访问子类中特有的成员（方法和属性）==。这是因为编译器在编译时只能识别引用类型（即父类类型）的成员，而无法识别子类特有的成员。只有在运行时通过动态绑定，才能调用子类中重写的成员方法
>
> *  可访问的成员
>   1. **父类中的方法和属性**：可以访问父类中定义的所有公共（`public`）、受保护（`protected`）和包私有（default）的方法和属性
>   2. **子类中重写的父类方法**：可以通过父类引用调用子类重写的方法，体现多态性
> *  不可访问的成员
>   1. **子类中特有的方法和属性**：不能直接通过父类引用访问子类特有的方法和属性
>   2. **子类中重载的方法**：重载方法属于编译时多态，需要参数匹配

```java
class Animal {
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }

    public void eat() {
        System.out.println("Animal is eating");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }

    public void fetch() {
        System.out.println("Dog is fetching");
    }

    public void eat(String food) { // 重载父类的eat方法
        System.out.println("Dog is eating " + food);
    }
}

public class Main {
    public static void main(String[] args) {
        Animal myDog = new Dog();   //向上转型

        // 可以访问父类的方法
        myDog.makeSound(); // 输出：Woof
        myDog.eat();       // 输出：Animal is eating

        // 不能访问子类特有的方法和属性
        // myDog.fetch(); // 编译错误：无法找到符号

        // 不能访问子类重载的方法
        // myDog.eat("bone"); // 编译错误：无法找到符号

        // 可以通过类型转换访问子类特有的方法和重载的方法
        if (myDog instanceof Dog) {
            Dog realDog = (Dog) myDog;   //向下转型
            realDog.fetch();       // 输出：Dog is fetching
            realDog.eat("bone");   // 输出：Dog is eating bone
        }
    }
}
```

> * 向上转型和向下转型分别对应于父类引用指向子类对象和强制类型转换
>
> * ### 向上转型（Upcasting）
>   
>   ==**向上转型**是将子类对象赋值给父类引用。因子类是父类的特殊化，因此这种转换是安全的、自动的且无需显式转换==
>   
>   #### 特点：
>
>   1. **自动进行**：无需显式转换
>   
>   2. **限制访问**：只能访问父类中定义的成员和子类重写的父类方法，不能访问子类特有的成员
> * ### 向下转型（Downcasting）
>
>   ==**向下转型**是将父类引用转换为子类引用（只能强转父类引用，不能强转父类对象，要求能转的父类引用必须是指向子类对象的）==。这种转换不是自动的，需要显式进行，并且必须确保转换是安全的，即父类引用实际指向的是子类对象。
>
>   #### 特点：
>
>   1. **显式进行**：需要显式转换。
>   2. **可能导致`ClassCastException`**：如果父类引用指向的不是目标子类对象，会抛出`ClassCastException`。
>   3. **扩展访问**：可以访问子类中特有的成员
>

```java
Animal animal = new Cat();
//下面的向下转型可以
Cat realcat = (Cat) animal;
//下面的向下转型会报错
Dog realdog = (Dog) animal;  //因为本来这个父类引用是指向Cat类，不能强转成其他类，只能转成Cat类
```
### 04.6.4 多态中属性的访问

> * ==属性的值，看编译类型==
> * 属性没有动态绑定机制

```java
public class Main {
    public static void main(String[] args) {
      Base b = new Sub();  //向上转型
      System.out.println(b.count);  //结果为10
    }
}

class Base{
    int count = 10;
}
class Sub extends Base{
    int count = 20;
}
```

### 04.6.5 instanceof 比较操作符

> * 用于判断对象的==运行类型==是否为XX类型或XX类型的子类型，这个操作符返回一个布尔值

```java
class Animal {
    // 父类的方法
}

class Dog extends Animal {
    // 子类的方法
}

public class Main {
    public static void main(String[] args) {
        Animal myAnimal = new Dog(); // 向上转型

        // 使用 instanceof 进行类型检查
        if (myAnimal instanceof Dog) {  //结果为true
            Dog myDog = (Dog) myAnimal; // 向下转型
            System.out.println("myAnimal 是 Dog 的实例");
        } else {
            System.out.println("myAnimal 不是 Dog 的实例");
        }
    }
}
```

```java
//处理多态
class Animal {
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }

    public void fetch() {
        System.out.println("Dog is fetching");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow");
    }
}

public class Main {
    public static void main(String[] args) {
        Animal[] animals = {new Dog(), new Cat(), new Animal()};

        for (Animal animal : animals) {
            animal.makeSound(); // 调用重写的方法

            // 使用 instanceof 确认对象类型并执行特定操作
            if (animal instanceof Dog) {
                Dog dog = (Dog) animal;
                dog.fetch();
            } else if (animal instanceof Cat) {
                System.out.println("This is a cat");
            } else {
                System.out.println("This is an animal");
            }
        }
    }
}
```

### 04.6.6 练习

```java
public class Main {
    public static void main(String[] args) {
        Sub s = new Sub();
        System.out.println(s.count);  //结果为20，访问属性看编译类型
        s.display();  //结果为20

        Base b = s;  //向上转型
        System.out.println(b == s);  //true，比较的是地址，两个引用都指向同一个对象
        System.out.println(b.count);  //结果为10
        b.display();  // 结果为20
    }
}

class Base{
    int count = 10;
    public void display(){
        System.out.println(this.count);
    }
}
class Sub extends Base{
    int count = 20;
    public void display(){
        System.out.println(this.count);
    }
}
```

### 04.6.7 动态绑定机制 **

> * Java的动态绑定机制，也称为晚期绑定或运行时多态，是面向对象编程（OOP）的一个特性，它允许根据运行时的实际对象类型来调用正确的方法，而不是根据编译时的声明类型
> * ==在Java中，当一个方法被调用时，Java虚拟机（JVM）会在运行时检查对象的实际类型，以确定哪个方法被调用。这与静态绑定相反，静态绑定是在编译时根据对象的声明类型来确定被调用的方法==
> * Java动态绑定的详细步骤：
>   1. **方法重写**：子类提供一个与父类中相同的方法，但具有不同的实现。子类中的方法具有相同的名称、返回类型和参数列表。
>   2. **方法调用**：创建一个子类的对象，并通过父类类型的引用变量调用一个方法。
>   3. **动态绑定**：在运行时，JVM检查引用变量所指向的对象的实际类型。这是通过调用对象的`getClass()`方法来实现的，该方法返回对象的实际类。
>   4. **方法查找**：==JVM在子类中搜索一个与被调用的方法具有相同的名称、返回类型和参数列表的方法。如果找到，JVM就会调用子类中的方法。如果没有找到，JVM就会在父类及其祖先类中搜索一个与被调用的方法具有相同的名称、返回类型和参数列表的方法。==
>   5. **方法调用**：JVM调用步骤4中找到的方法，并传递必要的参数。

```java
class Animal {
    void sound() {
        System.out.println("动物发出声音");
    }
}

class Dog extends Animal {
    @Override
    void sound() {
        System.out.println("狗叫");
    }
}

public class Main {
    public static void main(String[] args) {
        Animal animal = new Dog(); // 向上转型
        animal.sound(); // 输出：狗叫
    }
}
```

> * 动态绑定使程序更加灵活，因为它允许您编写可以处理不同类的对象的代码，而不需要在编译时知道对象的确切类型
> * ==当你调用一个对象的方法，而该方法在子类中没有找到，但是在父类中找到时，动态绑定机制将会调用父类的方法。然而，如果父类的方法调用了另一个在子类中被重写的方法，动态绑定机制将会调用子类的重写方法==

```java
class Animal {
    int age = 10;

    void sound() {
        System.out.println("动物发出声音");
        eat();
    }

    void eat() {
        System.out.println("动物吃东西");
    }
}

class Dog extends Animal {
    int age = 5;

    @Override
    void eat() {
        System.out.println("狗吃东西");
    }
}

public class Main {
    public static void main(String[] args) {
        Animal animal = new Dog();
        animal.sound(); // 输出：动物发出声音，狗吃东西
        System.out.println(animal.age); // 输出：10
    }
}
```

### 04.6.8 多态数组

> * 多态数组是指一个数组可以存储不同类型的对象，而这些对象都属于同一个父类或接口。这种特性使得我们可以编写更灵活和通用的代码

```java
// Animal类
class Animal {
    void sound() {
        System.out.println("动物发出声音");
    }
}

// Dog类，继承自Animal
class Dog extends Animal {
    @Override
    void sound() {
        System.out.println("狗叫");
    }
}

// Cat类，继承自Animal
class Cat extends Animal {
    @Override
    void sound() {
        System.out.println("猫叫");
    }
}

public class Main {
    public static void main(String[] args) {
        // 创建一个Animal类型的数组
        Animal[] animals = new Animal[3];

        // 将Dog和Cat对象存储在数组中
        animals[0] = new Dog();
        animals[1] = new Cat();
        animals[2] = new Animal();

        // 遍历数组并调用sound()方法
        for (Animal animal : animals) {
            animal.sound();  //这里会有动态绑定机制，不同的对象类型会调用对应的sound()
        }
    }
}
```

> * ==如果要调用数组中某个对象的特定方法，可以借助instanceof 来进行判断是否要进行向下转型==

```java
class Animal {
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("汪汪汪~");
    }

    public void fetch() { //特有的方法
        System.out.println("Dog is fetching");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("喵喵喵~");
    }

    public void scratch() {  //特有的方法
        System.out.println("Cat is scratching");
    }
}

public class Main {
    public static void main(String[] args) {
        // 创建一个父类类型的数组，可以存储不同子类的对象
        Animal[] animals = new Animal[3];
        //向上转型
        animals[0] = new Dog();
        animals[1] = new Cat();
        animals[2] = new Animal();

        for (Animal animal : animals) {
            animal.makeSound(); // 动态绑定，根据实际对象类型调用方法

            // 使用 instanceof 进行类型检查，然后直接进行向下转型调用特定方法
            if (animal instanceof Dog) {
                ((Dog) animal).fetch(); // 调用 Dog 特有的方法
            } else if (animal instanceof Cat) {
                ((Cat) animal).scratch(); // 调用 Cat 特有的方法
            } else {
                System.out.println("This is a generic animal.");
            }
        }
    }
}
```

### 04.6.9 多态参数 *

> * 多态参数（Polymorphic Parameters）是指==在方法的参数中使用父类类型或接口类型，从而可以接受该父类的所有子类或该接口的所有实现类的对象==。这种机制利用了多态性，使得方法可以处理不同类型的对象，提高了代码的灵活性和可扩展性
>
> * ### 基本概念
>
>   1. **父类类型参数**：在方法中使用父类类型作为参数，可以接受该父类的所有子类对象
>   2. **接口类型参数**：在方法中使用接口类型作为参数，可以接受该接口的所有实现类对象
>
> *  ### 多态参数的优势
>
>   1. **代码复用**：通过统一的父类或接口处理不同类型的对象，减少代码重复
>   2. **灵活性**：方法可以接受多种类型的对象，增强了代码的适应性
>   3. **简化代码**：使用多态参数可以简化方法的定义和调用逻辑

```java
class Animal {
    public void makeSound() {
        System.out.println("Some generic animal sound");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }

    public void fetch() {
        System.out.println("Dog is fetching");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow");
    }

    public void scratch() {
        System.out.println("Cat is scratching");
    }
}

public class Main {
    public static void main(String[] args) {
        Animal myDog = new Dog();
        Animal myCat = new Cat();
        Animal genericAnimal = new Animal();

        // 调用使用多态参数的方法
        performAction(myDog);
        performAction(myCat);
        performAction(genericAnimal);
    }

    // 定义一个使用多态参数的方法
    public static void performAction(Animal animal) {
        animal.makeSound(); // 动态绑定，根据实际对象类型调用方法

        // 使用 instanceof 进行类型检查，然后直接进行向下转型调用特定方法
        if (animal instanceof Dog) {
            ((Dog) animal).fetch(); // 调用 Dog 特有的方法
        } else if (animal instanceof Cat) {
            ((Cat) animal).scratch(); // 调用 Cat 特有的方法
        } else {
            System.out.println("This is a generic animal.");
        }
    }
}
```

# 05 Object类

## 05.1 `equals()`和`==`运算符

> * `==`运算符既可以判断基本类型，也可以判断引用类型
> * ==如果判断基本类型，判断的是值是否相等；如果判断引用类型，判断地址是否相等，即判定是否是同一个对象==

```java
public class Main {
    public static void main(String[] args) {
        A a = new A();
        A b = a;
        System.out.println(a == b);  // true
        B bobj = a;  //向上转型
        System.out.println(a == bobj);  // true
    }
}

class B{}
class A extends B{}
```

> * `equals()`方法是Object类的一个方法，只能判断引用类型
> * 把鼠标选在方法上，按`ctrl + B`，可以查看方法的源代码
> * 默认情况下，`equals()`方法与`==`运算符的行为相同，因为`Object`类的`equals()`方法是基于引用比较的。然而，==大多数类（如`String`、`Integer`等）都会重写`equals()`方法，以便进行内容比较==

```java
public class Main {
    public static void main(String[] args) {
        String s1 = new String("hello");
        String s2 = new String("hello");
        String s3 = s1;

        // 比较内容
        System.out.println(s1.equals(s2)); // 输出：true
        System.out.println(s1.equals(s3)); // 输出：true
    }
}
```

## 05.2 重写`equals()`

> * 重写`equals()`方法，用于判断两个对象是否相等（属性值相等）

```java
class Person{
    private String name;
    private int age;
    private char gender;

    //重写equals()方法
    public boolean equals(Object obj){
        if(this == obj){  //若为同一个对象
            return true;
        }
        if(obj instanceof Person){
            Person p = (Person)obj;  //向下转型
            return this.name.equals(p.name) && this.age == p.age && this.gender == p.gender;
        //上面name的equals方法是调用了String的equals方法
        }
        return false;
    }
}
```

## 05.3 练习

```java
public class Main {
    public static void main(String[] args) {
        int it = 65;
        float fl = 65.0f;
        System.out.println(it == fl);  //true

        char ch1 = 'A', ch2 = 12;
        System.out.println(ch1 == it);  //true
        System.out.println(ch2 == 12);  //true
    }
}
```

## 05.4 `hashCode()`方法

> * 可以提高具有哈希结构的容器的效率
> * 两个引用，如果指向同一个对象，则哈希值肯定一样；如果指向不同对象，哈希值是不一样的
> * 哈希值主要根据地址号来确定，但不能完全将哈希值等价于地址
> * `hashCode()`方法返回一个整数，这个整数是对象的哈希码。哈希码在哈希表（如`HashMap`、`HashSet`）等数据结构中用于快速查找。`hashCode()`方法是`Object`类的一部分，因此每个Java对象都有一个默认的哈希码

## 05.5 `toString()`方法

> * 主要作用是返回对象的字符串表示形式。默认情况下，`toString()` 方法返回对象的类名和它的内存地址的哈希码。这通常不是很有用，所以通常会重写 `toString()` 方法以提供更有意义的信息
> * ==在IDEA中用快捷键生成`alt + insert`,选择需要的属性输出==
> * ==当输出的为一个对象时，`toString()`方法会默认调用==

```java
class Person {
    private String name;
    private int age;
    private char gender;

    public Person(String name, int age, char gender) {
        this.name = name;
        this.age = age;
        this.gender = gender;
    }

    // 重写toString()方法
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + ", gender=" + gender + "}";
    }

    public static void main(String[] args) {
        Person p1 = new Person("John", 25, 'M');
        Person p2 = new Person("Jane", 30, 'F');

        // 直接打印对象
        System.out.println(p1); // 自动调用p1.toString()
        System.out.println(p2); // 自动调用p2.toString()
    }
}
```

## 05.6 `finalize()`方法

> * 用于在垃圾收集器（Garbage Collector）回收对象之前进行清理操作。它允许对象在被垃圾回收之前执行一些资源释放或清理工作，比如关闭文件、释放内存等
