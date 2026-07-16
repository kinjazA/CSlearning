# 01 断点调试 debug

> * ### 1. 设置断点
>
>   断点是指在代码中指定的一个位置，当程序运行到这个位置时会暂停，允许开发者检查程序的状态。
>
>   #### 步骤：
>
>   1. 打开要调试的项目。
>   2. 打开要调试的源代码文件。
>   3. 在左侧行号栏上点击要设置断点的行。
>      - 一个红色的圆点将出现在行号旁，表示断点已设置。
>      - 你可以通过再次点击红点来移除断点。
>
> * ### 2. 启动调试
>
>   #### 步骤：
>
>   1. 在 IDEA 的顶部菜单中，选择 "Run" -> "Debug"。
>      - 或者点击工具栏上的 Debug 按钮（通常是一个带有小虫子图标的按钮）。
>      - 也可以使用快捷键 `Shift + F9`。
>
> * ### 3. 调试控制面板
>
>   调试启动后，IDEA 将会打开调试控制面板，显示在窗口的下部。以下是主要的调试控制按钮：
>
>   - **Resume Program (F9)**: 恢复程序运行，直到下一个断点或程序结束。
>   - **Step Over (F8)**: 执行当前行，并停在下一行。如果当前行是一个方法调用，它会执行整个方法，而不会进入方法内部。
>   - **Step Into (F7)**: 进入方法内部。如果当前行是一个方法调用，它会进入方法内部进行逐步调试。
>   - **Step Out (Shift + F8)**: 跳出当前方法，返回调用该方法的位置，并停在那里。
>   - **Evaluate Expression (Alt + F8)**: 评估一个表达式，可以查看特定变量或表达式的值。
>   - **Stop (Ctrl + F2)**: 停止调试会话。
>
> *  ### 4. 查看变量和表达式
>
>   调试时，你可以查看当前变量的状态，评估表达式，以及检查堆栈帧：
>
>   - **变量窗口**：显示当前范围内的所有变量及其值。
>   - **堆栈帧**：显示当前线程的调用堆栈。你可以点击堆栈帧以查看不同调用层次的状态。
>   - **Watches**：你可以在 "Watches" 窗口中添加表达式，并实时查看这些表达式的值。
>
> * ### 5. 条件断点
>
>   有时你可能只希望在特定条件下暂停执行。这时可以使用条件断点。
>
>   #### 设置条件断点的步骤：
>
>   1. 右键点击已经设置的断点（红色圆点）。
>   2. 选择 "More" 或 "条件（Condition）"。
>   3. 输入条件表达式，例如 `i == 5`。只有当条件为真时，程序才会在此断点暂停。

![](day08_面向对象(高级).assets/image-20240731065450906.png)

> * `alt + shift + F7`强制进入键，进入方法体，看调用方法的具体实现细节
> * `shift + F8`可以逐层跳出方法

# 02 类变量（静态变量）

## 02.1 定义类变量

> * 类变量也称为静态变量（Static Variables）。它们属于类，==而不是类的任何特定实例对象，由该类所有对象共享。类变量在类加载时初始化，并且在整个程序运行期间只有一份拷贝==
> * 其实就是相当于一个变量捆着所有的对象，每个对象都能对其进行调用和修改
> * ==类变量使用`static`关键字定义==。它们通常被定义为`public`或`private`，以控制它们的访问级别

```java
public class MyClass {
    // 类变量（推荐写法）
    public static int classVariable = 10;
    //也有这样写的
	static public int classVariable = 21;
    
    // 实例变量
    public int instanceVariable = 20;
}
```
## 02.2 访问类变量

> * 类变量可以==通过类名直接访问==，顾名思义，是类的变量，而不需要实例化对象，==因为类变量是随着类的加载而创建的，所以没创建对象实例也可以访问修改==。也可以通过实例对象访问，但通常不推荐这样做

```java
public class Main {
    public static void main(String[] args) {
        // 通过类名访问类变量
        System.out.println(MyClass.classVariable); // 输出：10

        // 修改类变量
        MyClass.classVariable = 15;
        System.out.println(MyClass.classVariable); // 输出：15

        // 通过实例对象访问类变量（不推荐）
        MyClass obj = new MyClass();
        System.out.println(obj.classVariable); // 输出：15
    }
}

public class MyClass {
    // 类变量（推荐写法）
    public static int classVariable = 10;
}
```

## 02.3 类变量细节

> * 当需要让某个类的所有对象都共享一个变量时，考虑使用类变量
> * ==类变量和实例变量（成员变量、属性）的区别：类变量是所有对象共享的，实例变量是每个对象独有的==比方说要统计对象数量，==每个对象的变动都会影响这个值==，这种情况，设置一个类变量可以实现对象之间的共享，不容易出错
> * 类变量也要遵循访问修饰符的前提下进行访问
> * 类变量是在类加载时就初始化了的，即使没有创建对象，只要类加载了，就可以使用类变量。==类变量的生命周期是随着类的加载开始，随着类的消亡结束的==

# 03 类方法（静态方法）

## 03.1 定义类方法

> * 类方法也称为静态方法。==属于类，而不是类的任何特定实例。类方法在类加载时初始化，可以直接通过类名调用==
> * ==类方法使用`static`关键字定义==。它们通常被定义为`public`或`private`，以控制它们的访问级别

```java
public class MyClass {
    // 类方法
    public static void classMethod() {
        System.out.println("This is a class method.");
    }

    // 实例方法
    public void instanceMethod() {
        System.out.println("This is an instance method.");
    }
}
```

## 03.2 访问类方法

> * ==类方法可以通过类名直接访问，而不需要实例化对象。也可以通过实例对象访问==

```java
public class Main {
    public static void main(String[] args) {
        // 通过类名访问类方法
        MyClass.classMethod(); // 输出：This is a class method.

        // 通过实例对象访问类方法（不推荐）
        MyClass obj = new MyClass();
        obj.classMethod(); // 输出：This is a class method.
    }
}

public class MyClass {
    // 类方法
    public static void classMethod() {
        System.out.println("This is a class method.");
    }
}
```

```java
class Stu{
    private String name;
    private static  double fee = 0.0;

    public Stu(String name) {
        this.name = name;
    }

    public static void payFee(double fee){  //静态方法
        Stu.fee += fee;
    }

    public static void showFee(){  //静态方法
        System.out.println("共计收到学费"+ Stu.fee);
    }
}

public class Main {
    public static void main(String[] args) {
        Stu stu1 = new Stu("张三");
        stu1.payFee(100);  //通过创建对象来调用静态方法
        stu1.showFee();
        
        Stu stu2 = new Stu("李四");
        Stu.payFee(200);  //通过类名来调用静态方法
        Stu.showFee();
    }	
}
```

## 03.3 类方法细节

> * 当方法中不涉及到任何和对象相关的成员时，可以把方法设计成类方法（静态方法），提高开发效率
>
> * 工具类方法（Utility Methods）：==类方法非常适合实现不依赖于实例状态的功能。在这些情况下，方法仅依赖于输入参数，并返回计算结果==。例如，Java标准库中的`java.lang.Math`类就是一个工具类，所有的方法都是静态的
>
> * 类方法中没有this参数，super关键字也不能用；普通方法中隐含this参数
>
> * ==普通方法和对象有关，需要通过对象名调用；类方法可以通过类名调用，也可以通过对象名调用==
>
> * 类方法和普通方法都是随着类的加载而加载，将结构信息存储在方法区
>
> * ==类方法中只能访问静态变量（类变量）和静态方法（类方法）==
>
> * ==普通方法既可以访问普通变量\方法，又能访问静态变量\方法==

## 03.4 练习

```java
//判断下面代码的输出结果
public class Test {
    static int count = 9;
    public void count(){
        System.out.println("count=" + count++);  //符号在后，后加减
    }
    public static void main(String[] args) {
        new Test().count();  //9
        new Test().count();  //10
        System.out.println(Test.count);  //11
    }
} 
```

```java
//检查代码并改正，再判断输出结果
class Person{
    private int id;
    private static int total = 0;
    public Person(){
        total++;
        id = total;
    }
    public static int getTotalPerson(){
        id++;    //这里错了，静态方法不能访问非静态变量，删去该行
        return total;
    }
}

public class Test{
    public static void main(String[] args) {
        //没有创建对象，直接访问，结果为0
        System.out.println("数量为：" + Person.getTotalPerson()); 
        //创建对象了，构造器自动调用，total和id变为1，结果为1
        Person person = new Person();
        System.out.println("数量为：" + Person.getTotalPerson());  
    }
}
```

```java
//检查代码并改正，再判断输出结果，最终total等于多少
class Person{
    private int id;
    private static int total = 0;
    public Person(){
        total++;
        id = total;
    }
    public static void setTotalPerson(int total){
        this.total = total;  //这句出错，静态方法不能使用this关键字，删去该行
        Person.total = total;
    }
}

public class Test{
    public static void main(String[] args) {
        Person.setTotalPerson(3);
        new Person();
    }
}  //最终total值为4
```

# 04 main方法

## 04.1 main方法签名的说明

> * ==main方法是虚拟机调用的，JVM需要能够从外部访问这个方法，所以该方法的访问权限必须是public==
> * ==虚拟机在调用main方法时不必创建对象，所以该方法必须是static==
> * ==形参是一个String类型的数组，该数组包含了运行Java程序时传递给它的命令行参数==

```java
java MyClass arg1 arg2 arg3  
```

在我们自己编译java文件时，当编译好之后，执行.class文件时输入的命令行，后面可以追加参数，也就是像上面的MyClass 后面的arg，都会当做参数传入这个String类型的数组

> * 在一个类中只能有一个标准签名的`main`方法，但可以在不同的类中定义多个`main`方法。JVM将调用您指定的类的`main`方法作为程序的入口点
> * `main`方法是Java程序的入口点：必须具有标准签名`public static void main(String[] args)`。

## 04.2 main方法动态传值

> * 在IDEA中执行命令行的操作

![](day08_面向对象(高级).assets/image-20240801151756541.png)

![](day08_面向对象(高级).assets/image-20240801152053384.png)

# 05 代码块

> * 代码块是用一对大括号 `{}` 括起来的一段代码。代码块可以用于不同的上下文，有助于控制变量的作用范围和生命周期

## 05.1 普通代码块

> * ==普通代码块在创建对象时，会被隐式的调用，new一个对象，就会调用一次==
> * ==如果只是使用类的静态属性或方法，普通代码块不会执行==

### 05.1.1 局部代码块

> * **应用场景**：（Local Code Block）
>   - **控制变量作用范围**：在方法内部使用局部代码块，可以让某些变量只在代码块内生效，减少变量的作用范围，避免变量冲突
>   - **组织代码逻辑**：有时为了更好地组织代码逻辑，可以将代码分块处理

```java
public class Main {
    public static void main(String[] args) {
        // 开始一个局部代码块
        {
            int tempVar = 10;
            System.out.println("tempVar: " + tempVar); // 输出：tempVar: 10
        }
        // tempVar超出作用范围，以下代码会编译错误
        // System.out.println(tempVar);
    }
}
```

### 05.1.2 实例初始化代码块

> * **应用场景**：（Instance Initializer Block）
>   - **共享初始化逻辑**：当类有多个构造函数并且每个构造函数都需要执行相同的初始化代码时，可以将这些代码放到实例初始化块中，从而避免代码重复

```java
public class MyClass {
    // 实例初始化代码块
    {
        System.out.println("Initializing instance");
    }

    int instanceVar;

    public MyClass() {
        System.out.println("Constructor with no arguments");
    }

    public MyClass(int value) {
        System.out.println("Constructor with arguments");
    }

    public static void main(String[] args) {
        MyClass obj1 = new MyClass();
        MyClass obj2 = new MyClass(42);
    }
}
```

```java
//上面的代码输出结果为
Initializing instance
Constructor with no arguments
Initializing instance
Constructor with arguments
```

## 05.2 静态代码块

> * 静态代码块是由static和{}一块组成的，==随着类的加载而执行，并且只会被执行一次（也就是说如果new两个同类对象，静态代码也只会执行一次）==
> * ==类什么时候会被加载==：1. 创建对象实例时；2.创建子类对象实例时，父类会被加载（并且按照继承的机制，会先将父类中的静态代码块执行）；3.使用类的静态属性或方法时

## 05.3 类内部调用顺序 *

> * ==1.先是调用静态代码块和静态属性初始化，这两个优先级一样，如果同时存在，按照代码里的定义顺序执行==

```java
package com.codeblock;

public class CodeBlockDetail {
    public static void main(String[] args) {
        A a = new A();
    }
}

class A{
    private static int n1 = getN1();
    static {
        System.out.println("A的静态代码块");
    }

    public static int getN1() {
        System.out.println("getN1被调用");
        return 100;
    }
}
```

```java
//输出结果为
getN1被调用
A的静态代码块
```

> * ==2.调用普通代码块和普通属性的初始化，也是按照代码顺序来执行==

```java
package com.codeblock;

public class CodeBlockDetail {
    public static void main(String[] args) {
        A a = new A();
    }
}
class A{
    private int n2 = getN2();
    private static int n1 = getN1();

    {
        System.out.println("A的普通代码块");
    }

    static {
        System.out.println("A的静态代码块");
    }

    public static int getN1() {
        System.out.println("getN1被调用");
        return 100;
    }

    public int getN2() {
        System.out.println("getN2被调用");
        return 200;
    }
}
```

```java
//输出结果为
getN1被调用
A的静态代码块
getN2被调用
A的普通代码块
```

> * 3.==调用构造器==

```java
package com.codeblock;

public class CodeBlockDetail {
    public static void main(String[] args) {
        A a = new A();
    }
}
class A{
    private int n2 = getN2();
    private static int n1 = getN1();

    public A() {
        System.out.println("A的构造器");
    }

    {
        System.out.println("A的普通代码块");
    }

    static {
        System.out.println("A的静态代码块");
    }

    public static int getN1() {
        System.out.println("getN1被调用");
        return 100;
    }

    public int getN2() {
        System.out.println("getN2被调用");
        return 200;
    }
}
```

```java
//代码输出结果为
getN1被调用
A的静态代码块
getN2被调用
A的普通代码块
A的构造器
```

## 05.4 反编译看代码块细节 * 难

> * ==构造器里最前面其实隐含了`super()`和调用普通代码块==，可以通过查看编译文件反编译之后，会发现==java虚拟机会把普通代码块嵌入每个构造器的最前面==。而静态代码块则在类加载时执行，仅执行一次
> * ==IDEA中`ctrl + F9`可以对java文件进行编译成class文件==

```java
package com.codeblock;

public class CodeBlockDetail {
}
class A{
    public A() {
        System.out.println("A的无参构造器");
    }

    public A(int n){
        System.out.println("这是A的有参构造器");
    }

    {
        System.out.println("A的普通代码块");
    }
}
```

这段代码编译成`.class`文件之后，再反编译回`.java`文件。可以发现，java虚拟机会把普通代码块嵌入进每个构造器的最前面

![](day08_面向对象(高级).assets/image-20240801225656381.png)

> * 创建子类对象时，代码的调用顺序：（超级重要）
>
>   1.父类的静态代码块和静态属性（优先级一样，按定义顺序执行）
>
>   2.子类的静态代码块和静态属性（优先级一样，按定义顺序执行）
>
>   3.父类的普通代码块和普通属性（优先级一样，按定义顺序执行）
>
>   4.父类的构造器
>
>   5.子类的普通代码块和普通属性（优先级一样，按定义顺序执行）
>
>   6.子类的构造器

```java
package com.codeblock;

class Parent {
    // 父类静态变量
    static int parentStaticVar = initializeParentStaticVar();

    // 父类实例变量（普通属性）
    int parentInstanceVar = initializeParentInstanceVar();

    // 父类静态代码块
    static {
        System.out.println("父类静态代码块执行");
    }

    // 父类实例初始化块
    {
        System.out.println("父类实例初始化代码块执行");
    }

    // 父类构造器
    public Parent() {
        System.out.println("父类构造器被调用");
    }

    private static int initializeParentStaticVar() {
        System.out.println("父类静态变量初始化");
        return 1;
    }

    private int initializeParentInstanceVar() {
        System.out.println("父类实例变量（普通属性）初始化");
        return 1;
    }
}

class Child extends Parent {
    // 子类静态变量
    static int childStaticVar = initializeChildStaticVar();

    // 子类实例变量
    int childInstanceVar = initializeChildInstanceVar();

    // 子类静态代码块
    static {
        System.out.println("子类静态代码块执行");
    }

    // 子类实例初始化块
    {
        System.out.println("子类实例初始化代码块执行");
    }

    // 子类构造器
    public Child() {
        System.out.println("子类构造器被调用");
    }

    private static int initializeChildStaticVar() {
        System.out.println("子类静态变量初始化");
        return 1;
    }

    private int initializeChildInstanceVar() {
        System.out.println("子类实例变量（普通属性）初始化");
        return 1;
    }
}

public class CodeBlockDetail {
    public static void main(String[] args) {
        Child child = new Child();
    }
}
```

```java
//输出结果为
父类静态变量初始化
父类静态代码块执行
子类静态变量初始化
子类静态代码块执行
父类实例变量（普通属性）初始化
父类实例初始化代码块执行
父类构造器被调用
子类实例变量（普通属性）初始化
子类实例初始化代码块执行
子类构造器被调用
```

> * ==静态代码块只能调用静态成员，普通代码块可以调用类内任意成员==
>
> * 其实这里我非常疑惑，也就是编译器到底是如何执行，为什么代码块先执行，里面可以调用任意成员，比如说一个属性，但是属性不是要先声明再定义么，下面是GPT的解答
>
> * ### 编译器如何处理
>
>   编译器在处理Java代码时，会按以下步骤进行：
>
>   1. **解析变量声明**：编译器首先会解析类中所有变量的声明。无论这些变量在代码中的位置如何，它们的声明会在类的字节码中先于任何代码块或构造函数
>   2. **处理实例初始化块**：编译器会将实例初始化块的代码插入到每个构造函数的开头。因此，实例初始化块的代码在构造函数之前执行，但在执行时，这些实例变量已经声明了
>   3. **生成字节码**：编译器生成的字节码会确保变量的声明在所有初始化代码之前存在

```java
package com.codeblock;

public class CodeBlockDetail {
    // 实例初始化块
    {
        System.out.println("实例初始化代码块被执行");
        instanceVar2 = 100;
        instanceVar1 = 200;
    }

    // 实例变量
    int instanceVar1;
    int instanceVar2 = 50;

    // 无参数构造器
    public CodeBlockDetail() {
        System.out.println("构造器被调用");
    }

    public static void main(String[] args) {
        CodeBlockDetail obj = new CodeBlockDetail();
        System.out.println("构造器执行后的实例变量1:" + obj.instanceVar1);  //200
        System.out.println("构造器执行后的实例变量2:" + obj.instanceVar2);  //50
    }
}
```

反编译后的代码为：![](day08_面向对象(高级).assets/image-20240802093114751.png)

可以看到编译后，实例初始化代码块里的代码会嵌入构造器的最前面，并且是按照顺序的，然后才是构造器本身的一个初始化50

> * 下面我们把实例初始化代码块和实例变量（普通属性）的位置互换

```java
package com.codeblock;

public class CodeBlockDetail {
    // 实例变量
    int instanceVar1;
    int instanceVar2 = 50;
    // 实例初始化块
    {
        System.out.println("实例初始化代码块被执行");
        instanceVar2 = 100;
        instanceVar1 = 200;
    }

    // 无参数构造器
    public CodeBlockDetail() {
        System.out.println("构造器被调用");
    }

    public static void main(String[] args) {
        CodeBlockDetail obj = new CodeBlockDetail();
        System.out.println("构造器执行后的实例变量1:" + obj.instanceVar1);  //200
        System.out.println("构造器执行后的实例变量2:" + obj.instanceVar2);  //100
    }
}
```

![](day08_面向对象(高级).assets/image-20240802094251656.png)

因为普通属性和普通代码块的优先级一样，因此按照顺序执行。先初始化两个变量，一个为默认值，一个赋值50，然后执行普通代码块，默认值会改为200,50会被覆盖掉，写为100

# 06 单例设计模式（简述）

> * 是静态方法和属性的经典使用
> * 所谓类的单例设计模式，就是采取一定的方法保证在整个软件系统中，对某个类只能存在一个对象实例，并且该类只提供一个取得其对象实例的方法
> * 在Java中，实现单例模式有多种方法，包括懒汉式、饿汉式、双重检查锁定（Double-Checked Locking）、静态内部类（Bill Pugh Singleton Design）和枚举（Enum Singleton）等

## 06.1 饿汉式

> * 步骤：
>
>   1.私有化构造器
>
>   2.在类的内部创建一个私有对象
>
>   3.提供一个公开的静态方法访问这个私有对象
>
> * 只要类加载，对象就创建了
>
> * ==当第二次调用这个公开方法时，因为它返回这个对象时写成了静态变量的，在类加载时就创建了，而且只会执行一次，保证了是同一个对象==

```java
public class EagerSingleton {
    // 静态变量，保存唯一实例
    private static final EagerSingleton instance = new EagerSingleton();

    // 私有构造器，防止实例化
    private EagerSingleton() {
        // 可能包含一些初始化代码
    }

    // 静态方法，提供全局访问点
    public static EagerSingleton getInstance() {
        return instance;
    }

    public void showMessage() {
        System.out.println("Hello from EagerSingleton!");
    }

    public static void main(String[] args) {
        EagerSingleton singleton = EagerSingleton.getInstance();
        singleton.showMessage();
    }
}
```

## 06.2 懒汉式

> * 步骤：
>
>   1.构造器私有化
>
>   2.定义一个静态属性，类型就是这个对象，默认为null
>
>   3.提供一个公开的静态方法用于返回对象
> * 类加载，但是没有创建对象，只有在调用公开方法时才会创建对象
> * ==当第二次调用这个方法时，因为不为null了，所以不会新建对象，保持同一个对象==

```java
public class LazySingleton {
    // 静态变量，保存唯一实例
    private static LazySingleton instance;

    // 私有构造器，防止实例化
    private LazySingleton() {
        // 可能包含一些初始化代码
    }

    // 静态方法，提供全局访问点
    public static LazySingleton getInstance() {
        if (instance == null) {  //若没有创建对象
            instance = new LazySingleton();
        }
        return instance;
    }

    public void showMessage() {
        System.out.println("Hello from LazySingleton!");
    }

    public static void main(String[] args) {
        LazySingleton singleton = LazySingleton.getInstance();
        singleton.showMessage();
    }
}
```

> * **饿汉式单例模式**在类加载时就创建实例，确保实例在类加载时就可用，线程安全
>
>   #### 特点
>
>   1. **实例创建时机**：
>      - 在类加载时就创建实例
>      - 无论是否使用该实例，都会在类加载时创建
>   2. **线程安全**：
>      - 饿汉式单例模式在类加载时就创建实例，线程安全，不需要额外的同步机制
>   3. **实现简单**：
>      - 实现简单，代码易读
>   4. **资源利用**：
>      - 如果实例初始化开销大且不常用，会浪费系统资源

> * **懒汉式单例模式**在第一次调用 `getInstance` 方法时才创建实例，延迟加载实例
>
>   #### 特点
>
>   1. **实例创建时机**：
>      - 在第一次调用 `getInstance` 方法时创建实例
>      - 实现了延迟加载，只有在需要时才创建实例
>   2. **线程安全性**：
>      - 基础的懒汉式单例模式不是线程安全的，在多线程环境中可能会创建多个实例
>      - 需要额外的同步机制来确保线程安全，如 `synchronized` 关键字或双重检查锁定
>   3. **实现复杂度**：
>      - 实现相对复杂，特别是在需要确保线程安全的情况下
>   4. **资源利用**：
>      - 高效，只有在需要时才创建实例，避免了资源浪费

# 07 final 关键字

> * 可以用来修饰类、方法和变量（包括成员变量、局部变量和方法参数）

## 07.1 使用场景1--final类

> * 当一个类不希望被继承时，可以用`final`关键字来修饰
> * 当一个类被声明为 `final` 时，表示该类不能被继承。这样做的目的是为了确保该类的实现不能被改变
> * 编译器和JVM可能对`final`类进行优化，因为它们知道没有其他类会继承自这些类

```java
public final class FinalClass {
    public void display() {
        System.out.println("This is a final class.");
    }
}

// 以下代码将导致编译错误
// public class SubClass extends FinalClass {
//     // 编译错误: 不能继承自最终类
// }
```
## 07.2 使用场景2--final方法

> * 当不希望父类的某个方法被子类重写（override）时，可以用`final`关键字修饰
> * 当一个方法被声明为 `final` 时，表示该方法不能被子类重写。这样做的目的是为了确保方法的行为不能被改变

```java
public class ParentClass {
    public final void display() {
        System.out.println("This is a final method.");
    }
}

public class SubClass extends ParentClass {
    // 以下代码将导致编译错误
    // @Override
    // public void display() {
    //     System.out.println("Cannot override a final method.");
    // }
}
```
## 07.3 使用场景3--final变量

### 07.3.1 final成员变量

> * 当一个成员变量被声明为 `final` 时，表示该变量在初始化之后不能被修改。==`final`成员变量必须在声明定义时或构造器中初始化==

```java
public class FinalVariableExample {
    private final int value;

    public FinalVariableExample(int value) {
        this.value = value; // 必须在构造器中初始化
    }

    public void display() {
        System.out.println("Value: " + value);
    }
}

// 以下代码将导致编译错误
// public void setValue(int newValue) {
//     this.value = newValue; // 编译错误: 不能为最终变量赋值
// }
```

### 07.3.2 final局部变量

> * 当一个局部变量被声明为 `final` 时，表示该变量在初始化之后不能被修改。`final`局部变量必须在声明定义时或稍后的代码中初始化

```java
public void method() {
    final int localVar = 10;
    System.out.println("Local Variable: " + localVar);

    // 以下代码将导致编译错误
    // localVar = 20; // 编译错误: 不能为最终变量赋值
}
```

### 07.3.3 final方法参数

> * 当一个方法参数被声明为 `final` 时，表示该参数在方法中不能被修改

```java
public void display(final int number) {
    System.out.println("Number: " + number);

    // 以下代码将导致编译错误
    // number = 20; // 编译错误: 不能为最终参数赋值
}
```

## 07.4 使用细节

> * `final`修饰的属性又叫常量，一般用全大写字母命名
> * ==`final`修饰的属性在定义时，必须赋初值，并且之后不能再修改，赋值可以在以下位置：1.定义时；2.在构造器中；3.在实例初始化代码块中==
> * ==若`final`修饰的属性是静态的，则初始化的位置只能是：1.定义时；2.在静态代码块中，不能在构造器中赋值==
> * 若类不是`final`类，但含有`final`方法，则该方法虽不能重写，但是可以被继承
> * 一般来说，若类是`final`的，则没必要再将方法修饰成`final`的
> * 构造器不能被`final`修饰
> * `final`往往和`static`结合使用，原因可以GPT问问

# 08 抽象类

## 08.1概述

> * 抽象类是用来表示概念的类，而不是用来创建具体对象的类。抽象类提供了一个模板，==定义了子类必须实现的方法，同时可以包含已经实现的方法和成员变量。抽象类可以包含抽象方法和具体方法==
> * 使用 `abstract` 关键字定义抽象类和抽象方法
> * ==`abstract`只能修饰类和方法，不能修饰属性或是其他的==
> * 抽象类可以没有`abstract`方法

```java
public abstract class Animal {
    // 抽象方法，没有方法体，子类必须实现
    public abstract void makeSound();

    // 具体方法，子类可以继承
    public void eat() {
        System.out.println("This animal is eating.");
    }
}

```

> * ==子类必须实现所有的抽象方法，否则子类也必须声明为抽象类（就直接加一个abstract在类签名）==

```java
public class Dog extends Animal {
    // 实现抽象方法
    @Override·
    public void makeSound() {
        System.out.println("Woof");
    }

    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.makeSound(); // 输出：Woof
        dog.eat(); // 输出：This animal is eating.
    }
}
```

> * **抽象方法**：没有方法体，只定义了方法签名，必须在子类中实现
> * 当一个类中存在抽象方法时，需要将该类声明为abstract类
> * **具体方法**：有方法体，子类可以继承或重写这些方法
> * ==抽象类不能被实例化==

```java
public abstract class Vehicle {
    public abstract void startEngine();
}

// 以下代码将导致编译错误
// Vehicle v = new Vehicle();
```

> * ==抽象类可以有构造函数，子类在实例化时会调用抽象类的构造函数==

```java
public abstract class Shape {
    private String color;

    public Shape(String color) {
        this.color = color;
    }

    public String getColor() {
        return color;
    }

    public abstract double area();
}

public class Circle extends Shape {
    private double radius;

    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }

    public static void main(String[] args) {
        Circle circle = new Circle("Red", 5.0);
        System.out.println("Color: " + circle.getColor()); // 输出：Color: Red
        System.out.println("Area: " + circle.area()); // 输出：Area: 78.53981633974483
    }
}
```

## 08.2 抽象类的细节

> * ==抽象方法不能使用`private`、`final`和`static`来修饰，因为这些关键字和重写相违背（而抽象方法一定要被子类重写）==
> * 因为`final`修饰之后的类不能被继承，那这样就和抽象类的用法矛盾了
> * `static`不能用来修饰抽象类，因为这两个概念在语义上是不兼容的。抽象类需要被实例化为子类对象，而`static`成员与实例无关

```java
public abstract class Employee{
    String name;
    int id;
    double salary;

    public Employee(String name, int id, double salary) {
        this.name = name;
        this.id = id;
        this.salary = salary;
    }

    public abstract void work();
}

class CommonEmployee extends Employee{
    public CommonEmployee(String name, int id, double salary) {
        super(name, id, salary);
    }

    public void work(){
        System.out.println("普通员工"+ this.name +"正在上班");
    }
}

class Manager extends Employee{
    private  double bonus;

    public Manager(String name, int id, double salary, double bonus) {
        super(name, id, salary);
        this.bonus = bonus;
    }

    public void work(){
        System.out.println("经理"+ this.name +"正在管理其他人员");
    }
}
```

```java
public class Test {
    public static void main(String[] args) {
        CommonEmployee commonEmployee = new CommonEmployee("jack", 111, 5000);
        commonEmployee.work();

        Manager manager = new Manager("tom", 222, 5000, 1000);
        manager.work();
        //也可以使用多态数组来储存对象，通过动态绑定机制来实现对应的方法
    }
}
```

```java
//输出结果
普通员工jack正在上班
经理tom正在管理其他人员
```

## 08.3 应用--模板设计模式

> * 模板设计模式（Template Method Pattern）是一种行为设计模式，它==定义了一个算法的骨架，将一些步骤延迟到子类中实现。模板方法允许子类在不改变算法结构的情况下重新定义算法的某些步骤==。模板设计模式通常使用抽象类来实现

> * 假设我们要实现一个数据处理器，有不同类型的数据源（如CSV文件和数据库）。我们使用模板设计模式来定义数据处理的步骤

```java
public abstract class DataProcessor {
    // 模板方法，定义算法的骨架
    public final void process() {
        readData();
        processData();
        saveData();
    }

    // 抽象方法，具体步骤由子类实现
    protected abstract void readData();
    protected abstract void processData();

    // 具体方法，可以选择重写
    protected void saveData() {
        System.out.println("Saving data to database.");
    }
}
```

```java
//子类一
public class CSVDataProcessor extends DataProcessor {
    @Override
    protected void readData() {
        System.out.println("Reading data from CSV file.");
    }

    @Override
    protected void processData() {
        System.out.println("Processing CSV data.");
    }

    // 可选地重写saveData方法
    @Override
    protected void saveData() {
        System.out.println("Saving CSV data to database.");
    }

    public static void main(String[] args) {
        DataProcessor processor = new CSVDataProcessor();
        processor.process();
    }
}
```

```java
//子类二
public class DatabaseDataProcessor extends DataProcessor {
    @Override
    protected void readData() {
        System.out.println("Reading data from database.");
    }

    @Override
    protected void processData() {
        System.out.println("Processing database data.");
    }

    // 继承默认的saveData方法，无需重写

    public static void main(String[] args) {
        DataProcessor processor = new DatabaseDataProcessor();
        processor.process();
    }
}
```

> * ### 模板设计模式的优点
>
>   1. **复用代码**：公共的算法骨架代码可以在抽象类中实现，避免了重复代码
>
>   2. **灵活性**：子类可以通过实现抽象方法或重写具体方法来定制算法的具体步骤
>
>   3. **易于维护**：算法的步骤定义在一个地方（抽象类），子类只需实现具体步骤，逻辑清晰且易于维护
>
> * ### 适用场景
>
>   1. **多个类具有相似的处理步骤**：例如，不同类型的文件处理、数据处理等
>
>   2. **公共行为被不同子类重复实现**：将公共行为上移到抽象类中，实现代码复用
>
>   3. **控制子类的行为**：通过模板方法控制算法的执行步骤，防止子类改变算法的整体结构

# 09 接口

> * 接口（`interface`）是一种引用类型，用于定义类应该实现的一组方法签名。接口提供了一种抽象化的方式来定义行为规范，允许类通过实现接口来保证特定的行为。接口与类之间的关系是实现（implements）关系，而不是继承关系。接口主要用于设计应用程序的API，使得不同类可以有一致的行为接口

## 09.1 定义接口

> * ==使用 `interface` 关键字定义接口，接口中的方法默认是 `public abstract`，可以省略这些修饰符==
> * 接口里可以写以下四种：

```java
public interface Animal {
    // 抽象方法，省略了修饰符
    void makeSound();

    // 默认方法（Java 8及以后）
    default void eat() {
        System.out.println("This animal is eating.");
    }

    // 静态方法（Java 8及以后）
    static void sleep() {
        System.out.println("This animal is sleeping.");
    }

    // 常量可以，静态变量不行
    public static final String CATEGORY = "Animal";
    static int n1 ;  //这句会报错，一定要定义好的
}
```

## 09.2 实现接口

> * ==使用 `implements` 关键字实现接口，类必须实现接口中所有的抽象方法==

```java
public class Dog implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }

    // 可以选择重写默认方法
    @Override
    public void eat() {
        System.out.println("Dog is eating.");
    }

    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.makeSound(); // 输出：Woof
        dog.eat(); // 输出：Dog is eating.

        // 调用接口的静态方法
        Animal.sleep(); // 输出：This animal is sleeping.

        // 访问接口的常量
        System.out.println(Animal.CATEGORY); // 输出：Animal
    }
}
```

## 09.3 接口的细节

> * ==接口中的抽象方法没有方法体，只定义了方法签名，必须由实现类提供具体实现==
> * 默认方法是Java 8引入的新特性，允许在接口中定义带有方法体的方法。实现类可以选择重写这些默认方法，==默认方法需要使用`default`关键字修饰==
> * 接口可以包含常量，这些常量默认是 `public static final` 的，只能是`final`
> * 静态方法是Java 8引入的新特性，允许在接口中定义静态方法。静态方法不能被实现类重写，只能通过接口名调用
> * ==接口不能实例化==
> * 一个普通类实现接口，就要把这个接口里的抽象方法都实现，也就是重写。IDEA中的快捷键是鼠标定位在普通类这，然后`alt + enter`

![](day08_面向对象(高级).assets/image-20240803133015771.png)

> * ==抽象类实现接口时，可以不实现接口的抽象方法==
> * ==接口的修饰符只能是`public`和默认，和顶级类的修饰符一样==
> * 接口中的常量属性同样是直接由`接口名.属性名`来访问
> * 一个接口不能继承其他的类，但是可以继承多个别的接口

```java
// 接口 Flyable
public interface Flyable {
    void fly();
}

// 接口 Swimmable
public interface Swimmable {
    void swim();
}

// 接口 SuperAbility 继承 Flyable 和 Swimmable
public interface SuperAbility extends Flyable, Swimmable {
    void superPower();
}
```

在这个例子中，`SuperAbility` 接口继承了 `Flyable` 和 `Swimmable` 接口，因此任何实现 `SuperAbility` 接口的类都必须实现这三个接口的方法

## 09.4 类的多接口

> * ==使用逗号分隔每个接口，实现多个接口==
> * 类必须提供接口中所有抽象方法的实现，如果两个接口包含同名的方法，实现类只需实现一次

```java
// 接口 Flyable
public interface Flyable {
    void fly();
}

// 接口 Swimmable
public interface Swimmable {
    void swim();
}

public class Duck implements Flyable, Swimmable {
    @Override
    public void fly() {
        System.out.println("Duck is flying.");
    }

    @Override
    public void swim() {
        System.out.println("Duck is swimming.");
    }

    public static void main(String[] args) {
        Duck duck = new Duck();
        duck.fly(); // 输出：Duck is flying.
        duck.swim(); // 输出：Duck is swimming.
    }
}

```

> * ==一个类实现多个接口后，可以被视为这些接口的类型，从而实现多态性==
> * 可以使用接口类型的引用来指向实现类的实例   09.7详讲

```java
public class Duck implements Flyable, Swimmable {
    @Override
    public void fly() {
        System.out.println("Duck is flying.");
    }

    @Override
    public void swim() {
        System.out.println("Duck is swimming.");
    }

    public static void main(String[] args) {
        Duck duck = new Duck();
        
        // 使用 Duck 类型
        duck.fly(); // 输出：Duck is flying.
        duck.swim(); // 输出：Duck is swimming.
        
        // 使用 Flyable 类型
        Flyable flyable = new Duck();
        flyable.fly(); // 输出：Duck is flying.
        // flyable.swim(); // 编译错误，Flyable 接口没有 swim 方法
        
        // 使用 Swimmable 类型
        Swimmable swimmable = new Duck();
        swimmable.swim(); // 输出：Duck is swimming.
        // swimmable.fly(); // 编译错误，Swimmable 接口没有 fly 方法
    }
}
```

## 09.5 练习

```java
//判断下面语法是否正确，输出什么
interface  A{
    int a = 23;
}

class B implements A{
}

public class CC{
    public static void main(String[] args) {
        B b = new B();
        System.out.println(b.a);  //23
        System.out.println(A.a);  //23
        System.out.println(B.a);  //23
    }
}
```

## 09.6 接口VS抽象类 *

|      特性      | 抽象类                               | 接口                                       |
| :------------: | ------------------------------------ | ------------------------------------------ |
|  **定义方式**  | `abstract`关键字                     | `interface`关键字                          |
|  **成员变量**  | 可以包含实例变量、常量、静态变量     | 只能包含常量（`public static final `的量） |
|    **方法**    | 可以包含抽象方法、具体方法、静态方法 | 在java8+的版本，同样支持这三种方法         |
|   **构造器**   | 可以有构造器                         | 不可以有构造器                             |
|   **多继承**   | 只能继承一个类（包括抽象类）         | 不能继承类，但可以继承多个接口             |
| **访问修饰符** | 可以使用所有的访问修饰符             | 方法默认是 `public`，不能有其他访问修饰符  |
|  **使用场景**  | 需要定义类的部分实现并让子类共享时用 | 需要定义行为规范而不关心具体实现时用       |
| **能否实例化** | 不能实例化，但可以通过子类实例化     | 不能（匿名内部类可以）                     |

> * ### 抽象类和接口的联系
>
>   1. **抽象性**：抽象类和接口都可以用于定义抽象方法，需要子类或实现类提供具体实现
>   2. **多态性**：==抽象类和接口都可以用于实现多态性，通过父类引用或接口引用指向子类对象或实现类对象==
>   3. **设计目的**：都用于定义类应遵循的行为规范，但实现方式不同

> * 在实际开发中，可以结合使用抽象类和接口。例如，定义一个抽象类作为基类，包含一些共有的实现，同时使用接口定义类需要实现的行为

```java
// 接口 Flyable
public interface Flyable {
    void fly();
}

// 接口 Swimmable
public interface Swimmable {
    void swim();
}

// 抽象类 Animal
public abstract class Animal {
    private String name;

    public Animal(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public abstract void makeSound();
}

// 类 Duck 继承抽象类 Animal 并实现接口 Flyable 和 Swimmable
public class Duck extends Animal implements Flyable, Swimmable {
    public Duck(String name) {
        super(name);
    }

    @Override
    public void fly() {
        System.out.println(getName() + " is flying.");
    }

    @Override
    public void swim() {
        System.out.println(getName() + " is swimming.");
    }

    @Override
    public void makeSound() {
        System.out.println("Quack");
    }

    public static void main(String[] args) {
        Duck duck = new Duck("Daisy");
        System.out.println("Name: " + duck.getName()); // 输出：Daisy
        duck.makeSound();                             // 输出：Quack
        duck.fly();                                   // 输出：Daisy is flying.
        duck.swim();                                  // 输出：Daisy is swimming.
    }
}
```

## 09.7 接口的多态 *

> * 接口是实现多态性的关键机制之一。多态性允许对象以多种形式出现，是面向对象编程（OOP）的基本特性之一。通过接口实现多态，可以编写更加灵活和可扩展的代码。接口中的多态性通过接口引用指向实现类的对象，从而实现对不同类的统一操作

> * ### 接口中的多态实现
>
>   1. **定义接口**：接口中声明了多个方法，具体的实现类将提供这些方法的实现
```java
public interface Animal {
    void makeSound();
    void move();
}
```
>   2. **实现接口**：==不同的类实现同一个接口，提供具体的实现==
```java
// Dog 类实现 Animal 接口
public class Dog implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof");
    }

    @Override
    public void move() {
        System.out.println("Dog is running");
    }
}

// Cat 类实现 Animal 接口
public class Cat implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow");
    }

    @Override
    public void move() {
        System.out.println("Cat is walking");
    }
}
```
>   3. **接口引用**：==使用接口类型的引用来指向实现类的对象，从而实现多态性==
```java
public class Main {
    public static void main(String[] args) {
        // 使用接口引用指向实现类对象
        Animal myDog = new Dog();
        Animal myCat = new Cat();

        // 多态性 - 接口引用调用方法
        myDog.makeSound();  // 输出：Woof
        myDog.move();       // 输出：Dog is running

        myCat.makeSound();  // 输出：Meow
        myCat.move();       // 输出：Cat is walking

        // 可以将实现类对象存储在接口类型的数组中
        Animal[] animals = { myDog, myCat };

        // 遍历数组，调用接口方法，实现多态
        for (Animal animal : animals) {
            animal.makeSound();
            animal.move();
        }
    }
}
```
>   4. **动态绑定**：在运行时，根据引用指向的具体对象，调用对应的方法实现
>
>      在 `Main` 类中，使用 `Animal` 类型的引用指向 `Dog` 和 `Cat` 对象，通过接口引用调用 `makeSound` 和 `move` 方法，实现了多态性。

> * ### 多态的优势
>
>   1. **代码灵活性**：通过接口实现多态，可以编写更加灵活的代码，方便后续的扩展和维护
>   2. **统一接口**：通过接口定义一组通用的方法，不同的实现类提供具体的实现，增强代码的可读性和可维护性
>   3. **动态绑定**：在运行时，根据对象的实际类型调用相应的方法，实现了动态绑定，增强了程序的灵活性
>
>   ### 实际应用场景
>
>   1. **API设计**：接口用于定义API的行为规范，不同的实现类可以提供不同的实现
>   2. **框架设计**：框架设计中，接口用于定义扩展点，允许用户自定义实现
>   3. **多态集合**：通过接口实现多态集合，例如将不同类型的对象存储在同一个集合中，并统一处理

# 10 内部类 *

> * 内部类（Inner Class）是定义在另一个类内部的类。内部类提供了更好的封装性和组织性，可以访问其外部类的成员变量和方法
>
> * ### 内部类的类型
>
>   1. **成员内部类（Non-static Inner Class）**
>   2. **静态内部类（Static Inner Class）**
>   3. **局部内部类（Local Inner Class）**
>   4. **匿名内部类（Anonymous Inner Class**

## 10.1 成员内部类
> * 成员内部类是在外部类的成员位置定义的类。它可以访问外部类的所有成员变量和方法，包括私有成员
> * ==通过外部类的实例创建内部类的实例==

```java
public class OuterClass {
    private String outerField = "Outer field";

    public class InnerClass {
        public void display() {
            // 访问外部类的成员变量
            System.out.println("Outer field: " + outerField);
        }
    }

    public static void main(String[] args) {
        OuterClass outer = new OuterClass();
        OuterClass.InnerClass inner = outer.new InnerClass();
        inner.display(); // 输出：Outer field: Outer field
    }
}
```

## 10.2 静态内部类
> * 静态内部类使用 `static` 关键字定义，不能直接访问外部类的实例成员，只能访问外部类的静态成员
> * 可以直接通过外部类名创建静态内部类的实例

```java
public class OuterClass {
    private static String staticOuterField = "Static outer field";

    public static class StaticInnerClass {
        public void display() {
            // 访问外部类的静态成员变量
            System.out.println("Static outer field: " + staticOuterField);
        }
    }

    public static void main(String[] args) {
        OuterClass.StaticInnerClass inner = new OuterClass.StaticInnerClass();
        inner.display(); // 输出：Static outer field: Static outer field
    }
}
```

## 10.3 局部内部类
> * 局部内部类定义在方法或代码块内部，类似于局部变量。它只能在定义它的方法或代码块中使用
> * 只能在 `outerMethod` 方法内部使用

```java
public class OuterClass {
    public void outerMethod() {
        class LocalInnerClass {
            public void display() {
                System.out.println("Local Inner Class");
            }
        }

        LocalInnerClass localInner = new LocalInnerClass();
        localInner.display(); // 输出：Local Inner Class
    }

    public static void main(String[] args) {
        OuterClass outer = new OuterClass();
        outer.outerMethod();
    }
}
```

## 10.4 匿名内部类
> * ==匿名内部类是没有名称的内部类，通常用来实现接口或继承一个类，只能创建一次实例，匿名内部类可以实现接口或继承普通类和抽象类==

```java
//基本语法
类或接口 匿名内部类 = new 类或接口(参数列表){
    类体
};
```

分号不能少

### 10.4.1 基于普通外部类的匿名内部类

```java
public class OuterClass {
    public void display() {
        System.out.println("外部类display方法");
    }

    public static void main(String[] args) {
        // 创建继承 OuterClass 的匿名内部类实例
        OuterClass obj = new OuterClass() {
            // 重写 display 方法
            @Override
            public void display() {
                System.out.println("匿名内部类重写display方法");
            }
        };

        obj.display(); // 输出：匿名内部类重写display方法
    }
}
```

### 10.4.2 基于抽象类的匿名内部类

```java
// 抽象类 Animal
abstract class Animal {
    // 抽象方法 makeSound
    public abstract void makeSound();

    // 具体方法 sleep
    public void sleep() {
        System.out.println("Animal is sleeping");
    }
}

public class Test {
    public static void main(String[] args) {
        // 创建继承 Animal 的匿名内部类实例
        Animal obj = new Animal() {
            // 实现抽象方法 makeSound
            @Override
            public void makeSound() {
                System.out.println("Anonymous Inner Class making sound");
            }
        };

        obj.makeSound(); // 输出：Anonymous Inner Class making sound
        obj.sleep();     // 输出：Animal is sleeping
    }
}
```

### 10.4.3 基于接口的匿名内部类

```java
// 定义接口 Runnable
interface Runnable {
    void run();
}

public class Test {
    public static void main(String[] args) {
        // 创建实现 Runnable 接口的匿名内部类实例
        Runnable obj = new Runnable() {
            // 实现 run 方法
            @Override
            public void run() {
                System.out.println("Anonymous Inner Class running");
            }
        };

        obj.run(); // 输出：Anonymous Inner Class running
    }
}
```

