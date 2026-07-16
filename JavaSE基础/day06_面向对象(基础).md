# 01 类与对象

## 01.1 类与对象初识

> * 类是一种数据类型，对象就是一个具体的实例
> * 属性是类的组成部分，从概念上看，==成员变量=属性=field==，一般是基本数据类型，也可以是引用类型
> * 属性的定义语法：`访问修饰符 属性类型 属性名;`
> * 访问修饰符是用来控制属性的访问范围的，有四种：`public,proctected,默认,private`，在面向对象中级详细讲解
> * ==`private`修饰符不可以用于顶级类（也就是最外层的），可以用于内部类==
> * ==属性如果不赋值，也有默认值，规则和数组一致==

```java
//创建对象
//先声明，再创建
Cat cat;
cat = new Cat();

//直接创建
Cat cat = new Cat();
```
```java
public class Object01{
    public static void main(String[] args) {
        //实例化对象cat1
        Cat cat1 = new Cat();
        cat1.name = "小白";
        cat1.age = 3;
        cat1.color = "白色";
        //实例化对象cat2
        Cat cat2 = new Cat();
        cat2.name = "小花";
        cat2.age = 12;
        cat2.color = "花色";

        //访问对象属性
        System.out.println("名字："+cat1.name
        +" 年龄："+cat1.age+" 颜色："+cat1.color);

        System.out.println("名字："+cat2.name
        +" 年龄："+cat2.age+" 颜色："+cat2.color);        
    }
}

class Cat{
    //属性
    String name;
    int age;
    String color;
}
```

> * String类型会存放在方法区

![](day06_面向对象(基础).assets/9f49204bbff78e864e6d8b5de02a15d.jpg)

## 01.2 对象的内存分配机制 *

> java内存的结构分析
>
> * ==栈：一般存放基本数据类型（局部变量）==成员变量会存放在堆中
> * ==堆：存放对象、数组等等==
> * ==方法区：常量池（常量，比如字符串）、类加载信息==

![](day06_面向对象(基础).assets/3fd21a3e94a4f2a9e34fbeab463d4fd.jpg)

> java创建对象的流程分析
>
> * 先加载类信息（属性和方法信息，只会加载一次）
> * 在堆中分配空间，默认初始化，再把地址赋给对象
> * 进行指定初始化属性等等

![](day06_面向对象(基础).assets/83a50cd062c7617c93d20d0819a28d1.jpg)

# 02 成员方法

## 02.1 快速入门

> * ==类里除了定义一些属性之外，还定义一些类的行为，这就需要借助成员方法来实现==

```java
//成员方法的定义
访问修饰符 返回值的数据类型 方法名 (形参){
    语句;
    return 返回值;  //可选
}
```

```java
public class Method01{
    public static void main(String[] args) {
        //创建Person对象
        Person p = new Person();
        //设置属性
        p.name = "张三";
        p.age = 18;
        //调用成员方法
        p.speak();
        p.cal01();
        p.cal02(234);  //这里传参234
        p.cal02(100);   //可以多次调用

        int result = p.cal03(20,47 );  //这里用result接收返回值
        System.out.println("两个数的和最终的和为" + result);
    }
}

class Person{
    //属性
    String name;
    int age;
    //成员方法
    public void speak(){
        System.out.println("我会说话");
    }

    public void cal01(){
        //计算1加到1000的和
        int sum = 0;
        for(int i = 0; i <= 1000; i++){
            sum += i;
        }
        System.out.println("1加到1000的最终的和为" + sum);
    }

    public void cal02(int n){
        //计算1加到n的和
        int sum = 0;
        for(int i = 0; i <= n; i++){
            sum += i;
        }
        System.out.println("1加到" + n + "的最终的和为" + sum);
    }

    public int cal03(int num1, int num2){  //int表示方法执行后返回值类型
        //计算两数之和
        int sum = num1 + num2;
        return sum;  //返回值
    }
}
```

## 02.2 方法调用机制

![](day06_面向对象(基础).assets/dfb3dabfadce5abd072f43a9c40a559.jpg)

## 02.3 成员方法细节

> * 一个方法最多只有一个返回值,若要返回多个值，可以使用数组

```java
public class Method01{
    public static void main(String[] args) {
        //一次性返回多个值，借助数组实现
        Cal01 c1 = new Cal01();  //实例化
        int[] arr = c1.getSumandSub(10, 20);  //调用
        System.out.println("和为："arr[0]);
        System.out.println("差为："arr[1]);
    }
}

class Cal01{
    public int[] getSumandSub(int n1, int n2){
        int sum = n1 + n2;
        int sub = n1 - n2;
        int[] arr = {sum, sub};  //用数组装两个结果
        return arr;
        //或者是arr[0] = n1+ n2;
    }
}
```

> * ==如果方法要求有返回数据类型，则方法体中最后的执行语句必须为`return 值/表达式`，而且要求返回值的类型必须和return的值类型一致或兼容==

```java
class Cal01{
    public double f1(){
        double a = 10.0;
        return a;
    }

    public double f2(){
        int a = 10;
        return a;  //int-->double，自动类型转换
    }

    public int f3(){
        double a = 10.0;
        return a;  //double-->int，不符合自动类型转换
    }
}
```

> * ==如果方法为void，则方法体中可以没有`return`语句，或者只写`return;`==
> * 方法名称使用驼峰命名法
> * ==一个方法可以有多个形参，也可以没有参数，用逗号间隔，形参数据 类型可以为任意数据类型==
> * ==实参和形参的类型要一致或兼容，个数、顺序必须一致==
> * ==方法体中不能嵌套定义方法==，里面可以输入、输出、变量、运算、分支、循环、方法调用等等
> * ==同一个类中的方法可以直接调用==

```java
public class Method01{
    public static void main(String[] args) {
        A a = new A();
        a.sayOK();
    }
}

class A{
    public void print(String str){
        System.out.println(str);
    }

    public void sayOK(){
        print("fewqf");  //同一个类中的方法直接调用
    }
}
```

> * ==跨类中的方法调用，需要通过对象名调用==，先创建对象，再调用方法
> * 跨类的方法调用和方法的访问修饰符相关，后面详细说明

```java
public class Method01{
    public static void main(String[] args) {
        A a = new A();
        a.m1();
    }
}

class A{
    public void print(String str){
        System.out.println(str);
    }

    public void m1(){
        //跨类调用成员方法
        B b = new B();  //先要创建B类的对象
        b.hi();   //再调用B类中的hi方法
    }
}

class B{
    public void hi(){
        System.out.println("B类中的hi方法被执行");
    }
}
```

## 02.4 练习

```java
//练习一，判断所给数是偶数还是奇数
public class Method01{
    public static void main(String[] args) {
        AA a = new AA();
        boolean result = a.OddorEven(23);
        System.out.println("这个数是偶数吗？" + result);
    }
}

class AA{
    public boolean OddorEven(int num1){
        // 判断num1是否是偶数
        if(num1 % 2 == 0){
            return true;
        }else{
            return false;
        }
        //上面代码等价于 return num1 % 2 != 0 ？ true; false; 
    }
}
```

```java
//按指定行数和列数还有字符样式打印一个矩阵式的字符串
public class Method01{
    public static void main(String[] args) {
        AA a = new AA();
        String result = a.print01(4,4,"#");
    }
}

class AA{
    public String print01(int n1, int n2, String str){
        for(int i = 0; i < n1; i++){
            for(int j = 0; j < n2; j++){
                System.out.print(str);  //print()方法实现输出不换行
            }
            System.out.println();  //每行输出之后换行
        }
        return "打印完成";
    }
}
```

# 03 成员方法传参机制 * 难

## 03.1 值传递的情况

> * ==调用方法时，会独立创建一个新的栈空间（准确地说，在现有的线程栈上创建一个新的栈帧），当方法执行完毕后，空间会自动收回销毁（对应的栈帧会被弹出）==
> * 对于基本数据类型（如int, double, boolean等），Java使用值传递。这意味着当方法被调用时，实际参数的值会被复制到形式参数中。==在方法内部对参数的任何修改都不会影响外部的实际参数==

```java
public class MethodParameter{
    public static void main(String[] args) {
        int a = 12, b = 23;
        AA aa = new AA();
        aa.swap(a, b);
        System.out.println("a=" + a + ",b=" + b);  // a=12,b=23
    }
}

class AA{
    public void swap(int a, int b){
        System.out.println("\n交换前\na=" + a + ",b=" + b);
        // 交换
        int temp = a;
        a = b;
        b = temp;
        System.out.println("交换后\na=" + a + ",b=" + b);
    }
}
```

## 03.2 引用传递的情况


> * 对于引用数据类型，==如果方法内部对参数进行了修改（例如，修改了对象的状态），会影响到外部的实际参数==。因为==方法接收的是对象引用的副本，但这个副本指向的是同一个对象==。因此，通过这个引用副本所做的任何修改都会反映到原始对象上

```java
public class MethodParameter{
    public static void main(String[] args) {
        B b = new B();
        int[] arr = {1, 2, 3};
        b.test100(arr);
    }
}

class B{
    public void test100(int[] arr){
        arr[0] = 200;  // 修改数组元素
        //遍历数组
        System.out.println("遍历test00的数组");
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");  //结果为200 2 3
        }
        System.out.println();
    }
}
```

> * ==在一个方法内部将参数重新赋值为 `null` 或者指向另一个对象==，这并不会影响这个方法外部的引用。这是因为重新赋值操作只改变了方法内部的引用副本，而不是外部的原始引用（不理解可以画内存分析）

```java
public class MethodParameter{
    public static void main(String[] args) {
        B b = new B();
        Person p = new Person();
        p.name = "kinjazA";
        p.age = 18;
        b.test200(p);
        System.out.println("main的p.age为" + p.age);  //结果为18
    }
}

class Person{
    int age;
    String name;
}

class B{
    public void test200(Person p){
        p = null;  //此操作仅影响方法内部的局部变量，不会改变传入对象本身的引用
//当这个方法被调用时，这个p是在他自己的栈空间的，前面链接的p对象不受影响
    }
}
```

```java
public class MethodParameter{
    public static void main(String[] args) {
        B b = new B();
        Person p = new Person();
        p.name = "kinjazA";
        p.age = 18;
        b.test200(p);  //调用test200方法，会在栈里开辟独立空间
        //test方法里又把这个独立空间里的p指向了一个新的person对象，并进行赋值属性
        System.out.println("main的p.age" + p.age);  //访问的是main的p，结果为18
    }
}

class Person{
    int age;
    String name;
}

class B{
    public void test200(Person p){
    p = new Person();
    p.age = 99;
    p.name = "tom";
    }
}
```

> * 对于引用类型，复制的是引用本身，即一个引用的副本，而不是它所指向的对象
>
> * **引用**就像是某个对象的“地址”或“指针”，告诉你这个对象在内存中的位置
>
>   1.想象你有一本书，书的封面上有一个标签，这个标签上写着书在图书馆的位置。这个标签就是“引用”
>   2.当你拿到这个标签（引用），你可以通过它找到并阅读这本书（对象）
>   
> * 引用的副本
>
>   1.原标签和复印的标签是两个不同的标签，但它们都指向图书馆中同一本书
>
>   2.朋友可以用这个标签去找书并在书上做标记（修改对象），这时候，无论你还是朋友，都能看到书上的标记，因为它们指向同一本书
>
> * **引用的对象**就是引用指向的实际对象，即上面例子中的那本书
>
> * ==所以在引用类型作为参数时，java也是值传递，这个值就是引用的地址，如果我们修改这个引用，也就是相当于修改书的标签，是不会影响其它外部标签的，外部标签仍然准确无误的指向原先那个对象。但是如果我们是基于这个引用副本，即复制的标签去对对象做出修改，外部标签也是指向这个对象的，因此会受到影响==

## 03.3 练习--克隆对象

```java
 public class MethodParameter{
    public static void main(String[] args) {
        Person p = new Person();
        p.age = 18;
        p.name = "张三";
        MyTools mt = new MyTools();  //创建mytools对象
        Person p2 = mt.CopyPerosn(p);  //调用其中的方法
        System.out.println(p2.age);  //结果为18
    }
}

class Person{
    int age;
    String name;
}

class MyTools{
    public Person CopyPerosn(Person p){
        //创建新对象
        Person p2 = new Person();
        p2.age = p.age;
        p2.name = p.name;  //把原来对象的属性赋给p2对象
        return p2;
    }
}
```

# 04 方法递归调用

## 04.1 递归执行机制

> * 递归就是方法自己调用自己，每次调用时传入不同的变量，有助于解决复杂问题，简化代码
> * 适用于诸如`f(n)=f(f(n-1),*)`这样的可以不断拆分的问题
> * ==递归调用会先深入到最底层（即`n`减到最小可能值），然后逐层返回并输出`n`的值，所以输出的顺序与递归调用的顺序相反==
> * 下面的代码首先传入4，进入if，再次调用test(3),又进入if，变成test(2)，不满足if，输出n=2,。之后test(2)结束，退到外层test(3)，剩最后一句打印n=3，再退出到test(4)，剩最后一句 ，打印n=4

  ```java
  public class Recursion{
      public static void main(String[] args) {
          T t1 = new T();
          t1.test(4);
      }
  }
  
  class T{
      public void test(int n){
          if(n > 2){
              test(n-1);  // 递归调用
          }
          System.out.println("n = " + n);
      }
  }
  ```

![](day06_面向对象(基础).assets/75cf7555dd7d0c1457372e2b2be4446.jpg)

![](day06_面向对象(基础).assets/c7a424ef2881851291d3016f2cda659.jpg)

![](day06_面向对象(基础).assets/0862ec330bbfec769101357b4869fbc.jpg)

## 04.2 递归应用1--阶乘

```java
public class Recursion{
    public static void main(String[] args) {
        T t1 = new T();
        int res = t1.factorial(5);
        System.out.println(res);  //结果为120
    }
}

class T{
    public int factorial(int n){
        if(n == 1){
            return 1;  
        }else{
            return n * factorial(n-1);  //阶乘的递归
        }
    }    
}
```

> * 每当执行一个方法时，java会创建一个新的栈空间
> * 方法的局部变量是独立的，不会相互影响
> * 若方法中使用的是引用类型变量（比如数组），就会共享该引用类型的数据
> * 递归必须向退出递归的条件逼近，否则就是无限递归，会报错
> * 当方法执行完毕或是执行到return语句，就会返回。谁调用就把结果返回给谁，同时该方法执行完毕

## 04.3 递归应用2--斐波那契数列

```java
//不用递归的写法
public class Recursion{
    public static void main(String[] args) {
        T t1 = new T();
        int[] res = t1.fibo(9);  //指定斐波那契数列的长度
        //打印斐波那契数列
        for(int i = 0; i < res.length; i++){
            System.out.print(res[i] + " ");
        }
    }
}

class T{
    public int[] fibo(int n){
        if(n == 1){
            return new int[]{1};
        }else if(n == 2){
            return new int[]{1,1};
        }else{
            int[] arr = new int[n];
            arr[0] = 1;
            arr[1] = 1;
            for(int i = 2; i < n; i++){
                arr[i] = arr[i-1] + arr[i-2];
            }
            return arr;
        }
    }    
}
```

```java
//使用递归的写法
public class Recursion{
    public static void main(String[] args) {
        T t = new T();
        System.out.println("第七位的斐波那契数列的数字是" + t.fibo(7));

    }
}

class T{
    public int fibo(int n){
        if(n==1||n==2){
            return 1;
        }else{
            return fibo(n-1)+fibo(n-2);  //递归调用
        }
    }    
}
```

## 04.4 递归应用3--猴子吃桃

> * 一只猴子，有一堆桃子，猴子每天吃掉一半，再吃一个，每天如此，到了第十天，还没吃，发现只有一个桃子了。问最初有多少个桃子

 ```java
 public class Recursion{
     public static void main(String[] args) {
         T t = new T();
         int n = 1;
         int sum = t.chitao(n);
         if(sum != -1){  //// 如果计算结果有效，打印桃子数量
             System.out.println("第"+n+"天有"+sum+"个桃子");  //结果为1534
         }
     }
 }
 
 class T{
     public int chitao(int n){
         //求第n天有几个桃子 ，规律就是前一天的桃子等于后一天的桃子加一再乘以二
         if(n == 10){
             return 1;
         }else if(n >= 1 && n <= 9){
             return (chitao(n+1) + 1) * 2;  //吃桃递归的规律
         }else{
             System.out.println("输入有误");
             return -1;  //随便写个返回
         }
     }
 }
 ```

## 04.5 递归应用4--老鼠出迷宫

P222至P226

# 05 方法重载 **

> * ==java中允许同一个类中有多个同名方法存在，但要求形参列表不一致==,==对返回类型不做要求==
> * 参数列表不同可以是==参数的数量不同==，也可以是==参数的类型不同==，或者是==参数的顺序不同==。方法重载是实现多态性的一种方式

```java
public class OverloadingExample {
    // 重载方法：不同数量的参数
    public void display(int i) {
        System.out.println("Display int: " + i);
    }

    public void display(int i, double d) {
        System.out.println("Display int and double: " + i + ", " + d);
    }

    // 重载方法：不同类型的参数
    public void display(double d) {
        System.out.println("Display double: " + d);
    }

    // 重载方法：参数顺序不同
    public void display(double d, int i) {
        System.out.println("Display double and int: " + d + ", " + i);
    }

    // 主方法
    public static void main(String[] args) {
        OverloadingExample example = new OverloadingExample();
        example.display(5);            // 调用 display(int i)
        example.display(3.14, 2);      // 调用 display(int i, double d)
        example.display(3.14);         // 调用 display(double d)
        example.display(3.14, 2);      // 调用 display(double d, int i)
    }
}
```

> * 当调用一个重载的方法时，Java编译器使用以下步骤来确定要调用的确切方法：
>   1. **寻找匹配**：==编译器寻找与调用处提供的参数类型完全匹配的方法==
>   2. **类型提升**：如果没有找到完全匹配的方法，编译器会尝试==通过类型提升==（例如，将 `int` 提升为 `long`）来找到匹配的方法
>   3. **可变参数**：如果类中定义了可变参数的方法，并且没有找到其他匹配的方法，编译器会尝试将调用匹配到可变参数的方法
>   4. **最具体的方法**：==如果有多个方法都符合条件，编译器会选择最具体的方法（即参数类型最接近实际传递的参数类型的方法）==
>   5. **错误**：如果编译器无法找到合适的方法，它将抛出一个编译错误

```java
public class OverLoad{
    public static void main(String[] args) {
        Methods me = new Methods();
        int a = me.max(2313,543);
        double b = me.max(4312.514, 136.6134);
        int c = me.max(2313,543,123);
        System.out.println(a);
        System.out.println(b);
        System.out.println(c);
    }
}

class Methods{
    public int max(int n1,int n2){
        return n1 > n2 ? n1 : n2;  //三元运算符
    }

    public double max(double n1,double n2){
        return n1 > n2 ? n1 : n2;
    }

    public int max(int n1,int n2,int n3){
        return max(max(n1,n2),n3);  //调用max(int n1,int n2)
        //或者
        //int max1 = n1 > n2 ? n1 : n2;
        //return max1 > n3 ? max1 : n3;
    }
}
```

# 06 可变参数

> * java允许将==同一个类中的多个同名同功能但参数个数不同的方法封装成一个方法==，这些参数在方法内部被当作数组处理
> * **可变参数（Varargs）** 是一种特殊的参数，它允许你传递零个或多个参数给方法。==可变参数是通过在参数类型后面添加三个点`...`来声明的==
> * ==使用可变参数可以简化代码，特别是当需要编写一个能够处理不同数量参数的方法时==

```java
//基本语法
访问修饰符 返回类型 方法名(数据类型... 形参名){  
}
```

```java
public class VarParameters{
    public static void main(String[] args) {
        Methods me = new Methods();
        int a = me.sum(432,6243,7365,8567,6542);
        System.out.println(a);
    }
}

class Methods{
    public int sum(int n1,int n2){
        return n1 + n2;
    }

    public int sum(int n1,int n2,int n3){
        return n1 + n2 + n3;
    }
    //...
    //上面这样写代码过于啰嗦，可以借助可变参数的方式完成上面的需求
    public int sum(int... nums){
        int res = 0;
        for(int i = 0;i < nums.length;i++){
            res += nums[i];  //像数组那样访问
        }
        return res;
    }
}
```

> * 可变参数的实参可以是数组，即如果方法参数是可变参数，可以直接传递一个数组，而不需要解构数组

```java
public class VarParameters{
    public static void main(String[] args) {
        Methods me = new Methods();
        String[] greetings = {"Hi", "Bonjour", "Hola"};
        me.display(greetings);    // 直接传递一个数组
    }
}

class Methods{
    public void display(String... strings) {
        for (String str : strings) {
            System.out.println(str);
        }
    }
}
```

> * ==可变参数**必须**是方法参数列表中的最后一个参数==
> * 因为当编译器解析方法调用时，它需要能够明确地区分可变参数和其他参数。如果可变参数不是最后一个参数，编译器将无法确定哪些参数应该赋值给可变参数，哪些应该赋值给后续的普通参数
> * ==一个形参列表最多只能有一个可变参数==

```java
public class VarParameters{
    public static void main(String[] args) {
        Methods me = new Methods();
        me.display(3, "Hello", "World", "Java");
    }
}

class Methods{
    public void display(int count, String... strings) {
        System.out.println("Count: " + count);
        for (String str : strings) {
            System.out.print(str + " ");
        }
    }
}
```

```java
public class VarParameters{
    public static void main(String[] args) {
        Methods me = new Methods();
        double[] scores = {20,30,21.1};
        System.out.println(me.display("jack",scores));
    }
}

class Methods{
    public String display(String name, double... scores) {
        double sum = 0;
        for(int i = 0; i < scores.length; i++){
            sum += scores[i];
        }
        return name + "的总分为: " + sum;
    }
}
```

# 07 作用域

> * 在面向对象中，变量的作用域是非常重要的
> * 在java中，主要的变量就是属性（成员变量）和局部变量，局部变量一般是指在成员方法中定义的变量
> * ==局部变量只能在其定义的方法中使用，全局变量可以在其定义的整个类中使用，全局变量（属性）可以不赋值，前面讲过有默认值，后面再传入确切值。局部变量必须赋值后才能使用，因为没有默认值==

```java
public class VarScope{
    public static void main(String[] args) {
        Cat mimi = new Cat();
        mimi.cry();
        mimi.eat();
    }
}

class Cat{
    //全局变量
    //整个Cat类都可以访问age这个全局变量
    int age = 3;  
    public void cry(){
        //局部变量一般指在成员方法中定义的变量
        //这里的n和name都是局部变量
        int n = 0;
        String name = "miaomiao";
        System.out.println("miao miao miao");
        System.out.println(age);  //访问全局变量
    }

    public void eat(){
        System.out.println(age);  //访问全局变量
    }
}
```

> * Java中的变量可以根据它们被声明的位置被分类为以下几种作用域：
>   1. **局部变量（Local Variables）**：
>      - 声明在方法、构造函数或代码块内部
>      - 仅在声明它们的方法、构造函数或代码块中可见
>      - 当方法、构造函数或代码块执行完毕后，局部变量会被销毁
>      - 必须在使用前初始化，否则编译器会报错
>   2. **参数变量（Parameter Variables）**：
>      - 作为方法参数传递
>      - 仅在声明它们的方法中可见
>      - 方法调用结束后，参数变量将不再存在
>      - 必须在方法调用时提供值
>   3. **成员变量（Member Variables）**：
>      - 也称为实例变量，声明在类的内部，但在方法、构造函数或代码块之外
>      - 属于对象的一部分，每个对象实例都有其自己的成员变量副本
>      - 仅在创建对象后，并且对象在堆内存中存在时，成员变量才存在
>      - 可以不初始化，会有默认的初始化值
>   4. **静态变量（Static Variables）**：
>      - 也称为类变量，声明在类的内部，使用 `static` 关键字标记
>      - 属于类，而不是类的某个特定对象的实例
>      - 所有该类的实例共享同一个静态变量
>      - 存在于程序的整个执行期间
>      - 可以不初始化，会有默认的初始化值
>   5. **常量（Constants）**：
>      - 使用 `final` 关键字声明的变量
>      - 一旦被初始化，其值就不能被改变
>      - 可以是局部常量、成员常量或静态常量
>      - 常量的命名习惯是全部大写字母

```java
public class ScopeExample {
    // 成员变量
    private int instanceVariable;

    // 静态变量
    private static int staticVariable;

    public void myMethod() {
        // 局部变量
        int localVariable;

        // 常量
        final int constantVariable = 10;
    }

    // 参数变量
    public void anotherMethod(int parameterVariable) {
        // 方法体
        System.out.println("Parameter Variable: " + parameterVariable);
    }
}
```

> * ==全局变量（属性）和局部变量是可以重名的，访问时遵循就近原则==
> * ==在同一个作用域中，变量不能重名==，前面有提到过
> * ==全局变量可以跨类调用，具体有多种实现方法==，后面应该会讲

```java
public class VarScope{
    public static void main(String[] args) {
        Cat mimi = new Cat();
        mimi.cry();
    }
}

class Cat{
    //全局变量
    int age = 3;  
    public void cry(){
        int age = 4;
         // int age = 12; 会报错，同一个作用域，变量不能重名
        System.out.println(age);  //就近原则，输出4
    }

    public void eat(){
        int age = 5;
        System.out.println(age);  
        //就近原则，输出5,并且因为是不同的方法（作用域不同，因此变量可以重名）
    }
}
```

> * ### 作用域的规则：
>
>   - **更广泛的作用域**：变量的作用域越广泛，它的生命周期就越长
>   - **更狭窄的作用域**：变量的作用域越狭窄，它的生命周期就越短
>   - **变量隐藏**：在较窄的作用域中声明的变量可以隐藏（shadow）在较宽作用域中声明的同名变量
>   - **访问控制**：变量的访问权限由访问修饰符（如 `public`、`private`、`protected`）控制，==全局变量可以用访问修饰符控制，局部变量不可以==

# 08 构造方法/构造器

> * 是一种特殊的方法，==用于在创建对象时初始化对象==，没有创建对象不会调用构造器
> * 构造器的修饰符可以是默认的
> * `alt + insert`键可以生成构造器
> * ==构造器没有返回值==，不能写void返回
> * ==构造器方法名和类名必须一样==
> * 构造器的调用由系统完成，不需要调用，会自动完成对象的初始化
> * ==之前创建对象的写法是`new 类名();`，有了构造器之后，就可以`new 类名()`括号内可以传参，也就是指代对应的构造器，不传参，其实也是指代默认的构造器==

```java
//基本语法
访问修饰符 方法名(形参列表){
    方法体;
}
```

```java
public class Constructor{
    public static void main(String[] args) {
        Person p1 = new Person("jack",34);  //实例化对象
    }
}

class Person{
    String name;
    int age;
    public Person(String pname, int page) {  //构造方法
        System.out.println("构造方法被调用");  //会自动执行
        name = pname;
        age = page;
    }
    public void cry() {
        System.out.println("wawawa~");
    }
}
```

```java
public class Person {
    // 成员变量
    private String name;
    private int age;

    // 默认构造器
    public Person() {
        this.name = "Unknown";
        this.age = 0;
    }

    // 参数化构造器
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // 拷贝构造器
    public Person(Person anotherPerson) {
        this.name = anotherPerson.name;
        this.age = anotherPerson.age;
    }

    // 成员方法
    public void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }
}

public class Main {
    public static void main(String[] args) {
        // 使用默认构造器
        Person person1 = new Person();
        person1.display(); // 输出 Name: Unknown, Age: 0

        // 使用参数化构造器
        Person person2 = new Person("John Doe", 30);
        person2.display(); // 输出 Name: John Doe, Age: 30

        // 使用拷贝构造器
        Person person3 = new Person(person2);
        person3.display(); // 输出 Name: John Doe, Age: 30
    }
}
```

> * ==一个类可以创建多个构造器==
>
> * **构造器重载（Constructor Overloading）** 是指一个类中可以定义多个构造器，只要这些构造器的参数列表不同即可。构造器重载允许类以不同的方式创建对象，提供了灵活性和方便性。这是实现多态性的一种方式
>
> * ==如果没有定义构造器，系统会自动给类生成一个默认的无参数构造器==
>
> * ### 构造器重载的规则：
>  1. **构造器名称相同**：重载的构造器必须在同一个类中，并且具有相同的构造器名称（即类名）。
>   2. **参数列表不同**：参数的数量、类型或顺序必须至少有一个不同，以便编译器可以区分它们。
>   3. **返回类型**：构造器没有返回类型，因此返回类型不是区分构造器的因素。
>   4. **访问修饰符**：重载的构造器可以有不同的访问修饰符。
>   5. **抛出的异常**：重载的构造器可以抛出不同的异常。

```java
public class Car {
    private String brand;
    private int year;
    private double price;

    // 无参数构造器
    public Car() {
        this.brand = "Unknown";
        this.year = 0;
        this.price = 0.0;
    }

    // 带品牌和年份参数的构造器
    public Car(String brand, int year) {
        this.brand = brand;
        this.year = year;
        this.price = 0.0; // 默认价格
    }

    // 带品牌、年份和价格参数的构造器
    public Car(String brand, int year, double price) {
        this.brand = brand;
        this.year = year;
        this.price = price;
    }

    // 成员方法
    public void display() {
        System.out.println("Brand: " + brand + ", Year: " + year + ", Price: " + price);
    }
}

public class Main {
    public static void main(String[] args) {
        // 使用无参数构造器
        Car car1 = new Car();
        car1.display(); // 输出 Brand: Unknown, Year: 0, Price: 0.0

        // 使用带品牌和年份参数的构造器
        Car car2 = new Car("Toyota", 2020);
        car2.display(); // 输出 Brand: Toyota, Year: 2020, Price: 0.0

        // 使用带品牌、年份和价格参数的构造器
        Car car3 = new Car("Honda", 2018, 20000.0);
        car3.display(); // 输出 Brand: Honda, Year: 2018, Price: 20000.0
    }
}
```

# 09 对象创建流程分析

> ###  1. 类加载
>
> 当Java虚拟机（JVM）首次遇到对类 `Person` 的引用时，它将加载 `Person` 类。类加载过程包括：
>
> - **加载**：JVM读取 `Person` 类的 `.class` 文件，并创建一个 `java.lang.Class` 对象来表示这个类
> - **链接**：链接过程包括验证、准备和解析。在这个阶段，JVM会验证类文件的格式，为类的静态变量分配内存，并解析类中使用的符号引用
> - **初始化**：如果类中有静态初始化块或静态变量的初始化表达式，它们将在这个阶段执行
>
> ### 2. 分配内存
>
> JVM在堆内存中为新的对象分配内存。这个内存分配请求的大小基于 `Person` 类的大小，包括它的成员变量和对象的开销（如对象头信息）
>
> ### 3. 初始化默认值
>
> 在内存分配之后，JVM会将对象中的所有成员变量初始化为其类型的默认值。对于 `Person` 类：
>
> - `int` 类型的 `age` 被初始化为 `0`
> - `String` 类型的 `name` 被初始化为 `null`
>
> ### 4. 调用构造器
>
> 接下来，JVM调用 `Person` 类的构造器来初始化对象。构造器 `Person(String n, int a)` 被调用，并传入参数 `"jack"` 和 `20`
>
> #### 构造器执行步骤：
>
> - **设置 `name`**：构造器中的 `name = n;` 语句将参数 `"jack"` 赋值给成员变量 `name`
> - **设置 `age`**：构造器中的 `age = a;` 语句将参数 `20` 赋值给成员变量 `age`
>
> ### 5. 返回对象引用
>
> 构造器执行完毕后，JVM将新创建的对象的引用返回给变量 `p`。此时，`p` 指向堆内存中已初始化的 `Person` 对象
>
> ### 6. 使用对象
>
> 现在，`p` 变量持有对 `Person` 对象的引用，可以通过这个引用访问和操作对象的成员变量和方法

```java
class Person{
    int age = 90;
    String name;
    Person(String n, int a){
        name = n;
        age = a;
    }
}
Person p = new Person("jack", 20);
```

# 10 this关键字

> * 主要用来指代当前对象的引用，简单的说，哪个对象调用，this就代表哪个对象
> *  `this`关键字可以在对象的方法和构造函数中引用当前对象

```java
public class Person {
    private String name;
    private int age;

    public void setName(String name) {
        this.name = name; // this 指的是当前对象
    }

    public void setAge(int age) {
        this.age = age; // this 指的是当前对象
    }

    public void display() {
        System.out.println("Name: " + this.name + ", Age: " + this.age);
    }

    public static void main(String[] args) {
        Person person = new Person();  //这个person就是当前对象
        person.setName("Alice");
        person.setAge(30);
        person.display(); // 输出: Name: Alice, Age: 30
    }
}
```

![](day06_面向对象(基础).assets/463cf4eb9eed8728ba2e7fcd6347bfc.jpg)

> * 在构造函数或方法中，当局部变量名与实例变量名相同时，`this`关键字可以用来区分它们

```java
public class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name; // 使用 this.name 访问实例变量
        this.age = age;   // 使用 this.age 访问实例变量
    }

    public void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }

    public static void main(String[] args) {
        Person person = new Person("Bob", 25);
        person.display(); // 输出: Name: Bob, Age: 25
    }
}
```

> *  `this`关键字可以用来在一个构造器中调用同一个类的另一个构造器

```java
public class Person {
    private String name;
    private int age;

    public Person() {
        this("Unknown", 0); // 调用带两个参数的构造器
    }

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }

    public static void main(String[] args) {
        Person person = new Person();
        person.display(); // 输出: Name: Unknown, Age: 0
    }
}
```

> * 在方法中，`this`关键字可以用来返回当前对象的引用，以实现方法链调用

```java
public class Person {
    private String name;
    private int age;

    public Person setName(String name) {
        this.name = name;
        return this; // 返回当前对象
    }

    public Person setAge(int age) {
        this.age = age;
        return this; // 返回当前对象
    }

    public void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }

    public static void main(String[] args) {
        Person person = new Person();
        person.setName("Charlie").setAge(35).display();
        // 方法链调用，输出: Name: Charlie, Age: 35
    }
}
```

> *  `this`关键字可以作为参数传递给方法，以表示当前对象

```java
public class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    public void printPerson() {
        Printer.print(this); // 传递当前对象
    }

    public String getName() {
        return name;
    }

    public static void main(String[] args) {
        Person person = new Person("David");
        person.printPerson(); // 输出: Person's name: David
    }
}

class Printer {
    public static void print(Person person) {
        System.out.println("Person's name: " + person.getName());
    }
}
```

# 11 练习

P250至P262

