# 01 算术运算符

> * 对数值类型的变量进行运算
> * `+`,`-`,`*`,`/`,`%`,`++`(自增),`--`(自减)
## 01.1 除号,取模

> * ==当两个整数相除时,java的结果依然是整数(向下取整);若要保留小数,则应该把一个数写成小数形式==
> * ==因此为了避免`5/9`被java算成0,需要改写成`5.0/9`==
> * java中取模的本质公式为:`a % b = a - a / b * b`，这个和python的不一样，使用时，要仔细推敲

```java
public class Hello {
    public static void main(String[] args)  {
        //除号的使用
        System.out.println(10 / 4);  //结果为2
        System.out.println(10.0 / 4);  //结果为2.5
        double a = 10 / 4;  // 10/4结果为2，为int类型,又赋值给一个double类型
        System.out.println(a);  //结果为2.0

        //取余数(模)
        System.out.println(10 % 3);  //结果为1
        System.out.println(-10 % 3);  //结果为-1,和python不同,因为本质公式不一样
        System.out.println(10 % -3);  //结果为1
        System.out.println(-10 % -3);  //结果为-1
    }
} 
```

## 01.2 ++自增(--同理)

> * ==当独立使用时,前++和后++都完全等价于`i = i + 1`==
> * **在表达式使用时**,前++:先自增后参与运算;后++:先参与运算后自增
> * ==巧记:符号在前就先加/减，符号在后就后加/减==

```java
public class Hello {
    public static void main(String[] args)  {
        //++的单独使用
        int i = 10;
        i++;  // 等价于 i = i + 1,这里i就是11
        ++i;  // 等价于 i = i + 1,这里i是12
        System.out.println("i=" + i);  //结果为12

        //++作为表达式使用时   
        int j = 8;
        int k = ++j;  //前++,先自增后赋值,j = j + 1 , k = j ,这里k为9
        System.out.println("j=" + j + "k=" + k); //j=9,k=9
        int m = 8;
        int n = m++;  //后++,先赋值后自增,n = m,m = m + 1,这里n为8
        System.out.println("m=" + m + "n=" +n);  //m=9,n=8
    }
}
```

> * 感觉这里还是可以用巧记那个来理解
```java
public class Hello {  
    public static void main(String[] args)  {
        //代码分析,java中会使用临时变量,首先赋值temp=i,然后自增i=i+1,最后赋值i=temp
        int i = 1;
        i = i++;
        System.out.println(i);  //结果为1

        //代码分析,java中会使用临时变量,首先自增j=j+1,然后赋值temp=j,最后赋值j=temp
        int j = 1;
        j = ++j;
        System.out.println(j);  //结果为2
    }
}
```

```java
public class Hello {   
    public static void main(String[] args)  {
        int i1 = 10;
        int i2 = 20;
        //后加加,先赋值后运算
        int i = i1++;  //i为10,i1为11
        System.out.println("i=" + i);  
        System.out.println("i2=" + i2);
        //前减减,先运算后赋值
        i = --i2;  //i2为19,i为19
        System.out.println("i=" + i);
        System.out.println("i2=" + i2);
    }
} 
```

# 02 比较运算符

> * 比较两个数值或是变量的大小,结果得到一个boolean类型
> * ==在java中，char类型的单个字符可以用`==`来进行判断是否相等==
> * ==String类型的字符串不可以使用`==`来判断是否相等，而要使用`.equalsIgnoreCase()`（忽略大小写比较）或是`.equals（）`等方法==

```java
public class RelationalOperatorsExample {  
    public static void main(String[] args) {  
        int a = 10;  
        int b = 20;  
  
        // 等于  
        System.out.println(a == b); // 输出 false  
  
        // 不等于  
        System.out.println(a != b); // 输出 true  
  
        // 大于  
        System.out.println(a > b); // 输出 false  
  
        // 小于  
        System.out.println(a < b); // 输出 true  
  
        // 大于等于  
        System.out.println(a >= b); // 输出 false  
  
        // 小于等于  
        System.out.println(a <= b); // 输出 true  
    }  
}
```

```java
char a = 'A';
char b = 'A';
System.out.println(a == b); // 输出: true

String str1 = new String("Hello");
String str2 = new String("Hello");
System.out.println(str1 == str2); // 输出: false
System.out.println(str1.equals(str2)); // 输出: true
```

# 03 逻辑运算符

> * Java中的逻辑运算符主要用于根据一个或多个布尔表达式的值来执行逻辑运算，并返回布尔结果
> * 第一组:短路与`&&`,短路或`||`,逻辑非(取反)`!`
> * 第二组:逻辑与`&`,逻辑或`|`,逻辑异或`^`

## 03.1 短路与,逻辑与

> * 短路与和逻辑与在功能效果上是一样的,只是执行方式不同
> * ==逻辑与`&`无条件计算两边表达式，而短路与`&&`在左边表达式为`false`时不会计算右边表达式==
> * 逻辑与`&`更适用于位运算或需要同时考虑两边表达式结果的场景，而短路与`&&`更适用于需要提高执行效率的条件判断
> * 短路与`&&`由于能够避免不必要的计算，因此在执行效率上通常优于逻辑与`&`

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        // 短路与&& 和 逻辑与& 的使用
        int age = 50;
        if (age > 20 && age < 90) {  
            System.out.println("ok100");  
        }
        if (age > 20 & age < 90) {  
            System.out.println("ok200");
        }    
        //区别
        int a = 4;
        int b = 9;
        if (a < 1 && ++b < 50) {  //短路与如果第一个条件为false,后面条件不再执行
            System.out.println("ok300");      
        }
        System.out.println("a=" + a + ",b=" + b);  //短路与：a=4,b=9
        if (a < 1 & ++b < 50) {  //逻辑与条件都会执行
            System.out.println("ok400");      
        }
        System.out.println("a=" + a + ",b=" + b);  //逻辑与：a=4,b=10
    }  
}
```

## 03.2 短路或,逻辑或

> *  Java中的逻辑或`|`和短路或`||`在功能上相似，但在执行方式和效率上存在显著差异
> * ==逻辑或`|`会计算其左右两边的表达式，无论左边表达式的值是真（`true`）还是假（`false`）。即使左边表达式的值为`true`，右边表达式仍然会被计算==
> * ==短路或`||`采用短路方式执行。如果左边表达式的值为`true`，则不会计算右边表达式，因为整个表达式的值已经确定为`true`。只有当左边表达式的值为`false`时，才会计算右边表达式==
> * 由于短路或`||`能够避免不必要的计算，因此它在执行效率上通常优于逻辑或`|`

```java
public class LogicalOperator {  
    public static void main(String[] args) {  
        // 短路或|| 和 逻辑或| 的使用
        int age = 50;
        if (age > 20 || age < 30) {  
            System.out.println("ok100");  
        }
        if (age > 20 | age < 30) {  
            System.out.println("ok200");
        }    
        //区别
        int a = 4;
        int b = 9;
        if (a > 1 || ++b > 4) {  //短路或如果第一个条件为true,后面的条件不会执行
            System.out.println("ok300");      
        }
        System.out.println("a=" + a + ",b=" + b);  //短路与：a=4,b=9
        if (a > 1 | ++b > 4) {  //逻辑或条件都会执行
            System.out.println("ok400");      
        }
        System.out.println("a=" + a + ",b=" + b);  //逻辑与：a=4,b=10
    }  
} 
```

## 03.3 逻辑非(取反),逻辑异或

> * ==逻辑非就是把原先的表达式布尔类型给反一下==
> * ==逻辑异或是指当两个条件的布尔类型不同时,结果为true,否则为false==

```java
public class LogicalNotExample {  
    public static void main(String[] args) {  
        boolean isRainy = true;  
          
        // 使用逻辑非运算符取反  
        boolean isNotRainy = !isRainy;    
        System.out.println("Is it rainy? " + isRainy); // 输出：Is it rainy? true
        System.out.println("Is it not rainy? " + isNotRainy); // 输出：Is it not rainy? false  
          
        // 直接对字面量使用逻辑非  
        boolean isTrue = !false;  
        boolean isFalse = !true;  
          
        System.out.println("isTrue: " + isTrue); // 输出：isTrue: true  
        System.out.println("isFalse: " + isFalse); // 输出：isFalse: false  
    }  
}
```

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        // 逻辑异或^ 的使用public class LogicOperator {  
    public static void main(String[] args) {  
        // 逻辑异或^ 的使用
        boolean b = (10 > 1) ^ (3 < 6);
        System.out.println(b);  //结果为false
    }  
} 
        boolean b = (10 > 1) ^ (3 < 6);
        System.out.println(b);  //结果为false
    }  
}  
```

## 03.4 练习

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        int x = 5;
        int y = 5;
        //x++表示先使用x的值进行比较，然后x自增1
        //++y表示先使y自增1，然后使用新值进行比较
        if(x++ == 6 & ++y == 6){  //x不等于6,false;y等于6,true,整个表达式结果为false
            x = 11;
        }
        System.out.println("x=" + x + ",y=" + y);  //x=6,y=6       
    }  
}
```

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        int x = 5, y = 5;
        //x++表示先使用x的值进行比较，然后x自增1
        //++y表示先使y自增1，然后使用新值进行比较
        if(x++ == 6 && ++y == 6){  //前面为false,短路与,后面不执行
            x = 11;
        }
        System.out.println("x=" + x + ",y=" + y);  //x=6,y=5   
    }  
} 
```

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        int x = 5, y = 5;
        //x++表示先使用x的值进行比较，然后x自增1
        //++y表示先使y自增1，然后使用新值进行比较
        if(x++ == 5 | ++y == 5){  //前true,后false,整个表达式为true
            x = 11;
        }
        System.out.println("x=" + x + ",y=" + y);  //x=11,y=6   
    }  
}
```

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        int x = 5, y = 5;
        //x++表示先使用x的值进行比较，然后x自增1
        //++y表示先使y自增1，然后使用新值进行比较
        if(x++ == 5 || ++y == 5){  //前true,短路或,整个表达式为true
            x = 11;
        }
        System.out.println("x=" + x + ",y=" + y);  //x=11,y=5   
    }  
} 
```

# 04 赋值运算符

> * 就是将某个运算后的值,赋给指定的变量
> * 有基本的复制运算符`=`和复合赋值运算符`+=`,`-=`,`*=`,`/=`,`%=`
> * `a += b`等价于`a = a + b`,其余类似
> * ==复合赋值运算符会进行类型转换==

```java
public class LogicOperator {  
    public static void main(String[] args) {  
        byte b = 3;
        b += 2;  // 等价于 b = (byte)(b + 2);
        // b = b + 2;会报错,因为int不能转byte
        b++;  // 等价于 b = (byte)(b + 1);
    }  
} 
```

#  05 三元运算符 *

## 05.1 基本用法

> * ==基本语法:条件表达式 ? 表达式1 : 表达式2;==
> * ==如果条件表达式为true，三元运算符结果为表达式1；条件表达式结果为false，三元运算符结果为表达式2==
> * ==表达式1和2要为可以赋给接收变量的类型==(或是可以自动转换的,也就是不可以高精度赋给低精度),或者使用强制转换
> * 三元运算符可以转成if--else语句
> * ==整个三元运算符必须被看作一个整体来使用==
> * 嵌套可能会使代码难以阅读，因此在实际编程中应谨慎使用

```java
public class TernaryOperator {  
    public static void main(String[] args) {  
        int a = 5;  
        int b = 10;  
        int result = (a > b) ? a++ : b--;  
        System.out.println("结果是:" + result );  //result为10
        System.out.println("a的值是:" + a );  //a为5,没有执行
        System.out.println("b的值是:" + b );  //b为9,执行了
    }  
} 
```

```java
public class TernaryOperator {  
    public static void main(String[] args) {  
        int a = 5;  
        int b = 10;  
        //byte c = (a > b) ? a : b;会报错,高精度不能赋给低精度  
    }  
}
```

```java
public class TernaryOperator {  
    public static void main(String[] args) { 
        //判断三个数里的最大值
        //先两个里判断出较大的,再和第三个数比
        int a = 847658;  
        int b = 631234516; 
        int c = 311; 
        int max1 = a > b ? a : b; 
        int max = max1 > c ? max1 : c;
        System.out.println("max=" + max);
        
        //嵌套的写法
        int a = 5, b = 10, c = 15;  
        //首先比较ab，然后根据结果选择ac或bc之间的较大者进行比较，最终得到三个数中的最大值
		int max = (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);  
		System.out.println("三个数中最大的数是: " + max);
    }  
} 
```

## 05.2 三元运算符的类型提升

> * 在Java中，三元运算符的两个结果表达式（即`expression1`和`expression2`）必须返回兼容的类型，因为三元运算符的最终结果类型由这两个表达式共同决定
> * **编译器要求**：`expression1`和`expression2`必须具有相同的类型，或者其中一个可以通过自动类型转换（如类型自动提升）转为另一个类型
> * **类型自动提升**：当`expression1`和`expression2`的类型不同，但存在继承关系或基本类型与包装类型之间的转换关系时，编译器会尝试将它们提升为一个公共的父类型

```java
public class TernaryOperatorExample {
    public static void main(String[] args) {
        boolean condition = true;

        // 使用三元运算符，两个表达式返回不同的包装类型
        Number result = condition ? new Integer(10) : new Double(20.5);

        System.out.println("Result: " + result);
        System.out.println("Result type: " + result.getClass().getName());
    }
}
```

```java
//结果为
Result: 10.0
Result type: java.lang.Double
```

> * Java编译器在处理三元运算符时，会尝试找到两个类型的最接近的公共类型。在这个例子中，`Integer` 和 `Double` 的最接近公共类型是 `Double`，因为 `Double` 能够表示更广泛的数值范围
>
> * ### 三元运算符整体性的重要性
>
>   当我们说三元运算符是一个整体时，这意味着：
>
>   1. **类型统一**：整个三元运算符表达式必须能够统一成一个确定的类型，这个类型是由两个结果表达式共同决定的。
>   2. **编译时类型检查**：编译器会根据三元运算符的两个表达式来确定最终的返回类型，并确保返回的类型是有效的。如果两个表达式类型不兼容，编译器会报错。
>   3. **结果类型的确定**：三元运算符的返回值类型必须是两个表达式类型的一个共同超类型（通常是最低的公共父类）。在这种情况下，`Number`是`Integer`和`Double`的公共父类型，因此它们可以被自动提升并最终返回`Number`类型

# 06 运算优先级

> * 运算符的优先级决定了表达式中操作的执行顺序。当表达式包含多个运算符时，运算符的优先级决定了哪些操作首先执行
> * 在涉及优先级较高的运算符时使用括号`()`来明确指定操作的顺序，即使它们已经按照优先级正确地执行。这样做可以减少错误

# 07 标识符命名的规则与规范

> * 各种变量,方法,类名,即可以自己取名的都可算作标识符
>
> * 命名规则:
>
>   (1)由英文字母大小写,0~9,_,$组成
>
>   (2)数字不可以开头
>
>   (3)不使用关键字和保留字,但可以包含
>
>   (4)严格区分大小写
>
>   (5)不能包含空格

> * 命名规范:
>
>   (1)包名:多单词组成时,全小写,间隔用`.`隔开,例如`aaa.dsds.hdd`
>
>   (2)类名,接口名:多单词时,全部单词首字母大写,例如`NumCount`
>
>   (3)变量名,方法名:多单词时,第一个单词首字母小写,后面单词首字母大写,例如`tankShotGame`
>
>   (4)常量名:全大写,单词间隔用`_`隔开,例如`XXX_YYY_ZZZ`

# 08 关键字,保留字

> * 关键字指被java赋予了特殊含义,用做专门用途得分字符单词
> * 保留字是指现有java版本尚未使用,但以后版本可能作为关键字使用,尽量避开

# 09 键盘输入语句

> * 步骤:
>
>   (1)导入Scanner类所在的包
>
>   (2)创建该类对象(声明变量)
>
>   (3)调用里面的功能

```java
import java.util.Scanner;  //步骤一,把java.util包下的Scanner类导入
public class Input {  
    public static void main(String[] args) { 
        //步骤二,创建scanner对象实例(对象)
        Scanner sc = new Scanner(System.in);
        //步骤三,接收用户输入,使用相关方法
        System.out.println("请输入名字：");
        String name = sc.next();
        System.out.println("请输入年龄：");
        int age = sc.nextInt();
        System.out.println("请输入薪水：");
        double salary = sc.nextDouble();

        System.out.println("名字：" + name + "\n年龄："
         + age + "\n薪水：" + salary);
    }  
}
```

# 10 进制

> * 对于整数,有以下常用进制:
>
>   (1)二进制:0,1 ,以0b或0B开头表示
>
>   (2)十进制:0~9 ,
>
>   (3)八进制:0~7 ,以数字0开头表示
>
>   (4)十六进制:0~9及A(10)~F(15) ,以0x或0X开头表示(字母不区分大小写)

## 10.1 进制转换



# 11 原码,反码,补码



# 12 位运算符















