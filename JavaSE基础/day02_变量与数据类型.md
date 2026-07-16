# 01 变量概述

> * 使用的基本步骤:
>
>   ==(1)声明变量,就是说明这个变量的数据类型==
>
>   ==(2)变量赋值==

```java
public class Hello {
    public static void main(String[] args)  {
        //声明变量
        int a;
        a = 10;
        System.out.println(a);

        //还可以这样使用
        int b = 100;
        System.out.println(b);
    }
} 
```

> * 变量表示内存中的一个存储区域,==必须先声明,后使用==
> * ==在该区域的数据可以在同一数据类型范围内变化==
> * 变量 = 变量名 + 值 + 数据类型

```java
public class Hello {
    public static void main(String[] args)  {
        //变量先声明,后使用 
        int a = 100;
        System.out.println(a);
        //a = "hello"; 这是错的,不同数据类型
        a = 4213;
        // int a = 431; 这也是错的,在同一个作用域内变量不能重名
    }
```

# 02 程序中的+号

> * 当 + 号左右两边都是数值型时,做加法运算
> * ==当 + 号左右两边有一边为字符串（string类型，char类型不是）时,做拼接运算==
> * ==当对两个char类型的变量用 + 号时,是其对应的ASCII码值进行计算,而非字符拼接==
> * 运算顺序从左往右

```java
public class Hello {
    public static void main(String[] args)  {
        System.out.println(100 + 79); //结果为179
        System.out.println("erg" + 5324);  //结果为"erg5324"
        System.out.println("gaero" + 133 + 43);  //结果为"gaero13343"
    }
} 
```

# 03 数据类型

> * ==java每一种数据都要定义明确的数据类型,是强类型语言==
> * 分基本数据类型(整数,浮点,字符型,布尔型)和引用数据类型(类,接口,数组)

## 03.1 整型

> * 整数型:有byte(占1个字节,最大127),short(占2个字节,最大`2^15-1`),int(占4个字节,最大`2^31-1`),long(占8个字节,最大`2^63-1`)四种
> * ==long类型需要在后面加字母L==

## 03.2 浮点型

> * 浮点型:有float(占4个字节,单精度,通常是六七位小数的精度),double(占8个字节,双精度,通常是十五位小数的精度)两种
> * ==浮点数默认double类型,若要float类型,需要加f==

```java
public class Hello {
    public static void main(String[] args)  {
        int a = 188;
        //long类型需要在后面加字母L
        long b = 1L;  
        //浮点数默认double类型,若要float类型,需要加f
        float c = 1.0f;
        double d = 1.0;  
        
        //下面是错的,1.1默认是double类型,8字节,赋值给float类型,4字节,会出错
        //float h = 1.1;
        
        char e = 'a';
		char c1 = 68;  //打印出来是D
        boolean f = true;
        String g = "hello world";
    }
}
```
> * 当我们对运算结果为小数的进行相等判断时,要小心,因为浮点数都是近似值
 ```java
 public class Hello {
     public static void main(String[] args)  {
         double num1 = 2.7;
         double num2 = 8.1 / 3;
         System.out.println(num1);  //结果为2.7
         System.out.println(num2);  //结果为2.6999999999999997
     }
 }
 
 
 public class Hello {
     public static void main(String[] args)  {
         double num1 = 2.7;
         double num2 = 8.1 / 3;
         //正确的比数是看两数差值是否在符合需求的一个精度内,这里用0.0000001
         if (Math.abs(num1 - num2) < 0.0000001) {
             System.out.println("num1 equal to num2");
         }
     }
 } 
 ```

## 03.3 字符型

> * ==字符常量是**用单引号括起的**,双引号会被识别成字符串==
> * char的本质是一个整数,在输出时,是Unicode码对应的字符
> * ==char类型可以进行运算,会对应变化其Unicode码==
> * 字符型:char(占2个字节,**存放单个字符**),char类型也可以直接存放一个数,不过打印出来会发现并不是这个数,而是一个字符,这个数字实际是这个字符的编码
> * Unicode编码兼容ASCII编码,所以像英文字符是一致的
> * ==char类型一定是单引号括起的，String类型是双引号括起的，不能混==

```java
public class Hello {
    public static void main(String[] args)  {
        char c1 = 'a';
        char c2 = 'b';
        //输出'a'对应的数字
        System.out.println((int)c1);  //结果为97
        char c2 = 97;
        //输出97对应的字符
        System.out.println(c2);  //结果为a
        //char类型运算
        System.out.println('a' + 10);  //结果为107
        System.out.println(c1 + c2);  //结果也是对应的字符码值之和
        //前面提到的加号字符拼接是因为是字符串,这里是单个字符,所以是ASCII码进行运算
    }
} 
```

## 03.4 布尔型

> * 布尔型:boolean(占1个字节,存放true,false)
> * ==java里不可以用0或非0的整数来代替false和true(其他有的语言有这个特性)==

```java
public class Hello {
    public static void main(String[] args)  {
        //定义一个布尔类型变量用于逻辑判断
       boolean isPass = true;
       if(isPass == true) {
        System.out.println("考试通过");
        }else{
            System.out.println("考试不通过");
        }
    }
}
```

## 03.5 基本数据类型转换

> * ==在运算时,java会自动将精度小的类型转换成精度大的数据类型==
> * ==char-->int-->long-->float-->double==
> * ==byte-->short-->int-->long-->float-->double==
> * 从`long`（64位）到`float`（32位）的转换在一般情况下可能会涉及精度损失。如果`long`类型的值超出了`float`能够精确表示的范围（特别是非常大或非常小的数值），那么转换结果将是一个近似的浮点数，而不是原始的精确值

```java
public class Hello {
    public static void main(String[] args)  {
        int a = 'a';
        double b = 80;
        System.out.println(a);  //结果为97
        System.out.println(b);  //结果为80.0
    }
}  
```

> * 细节一:多种类型**混合运算时,会自动将所有数据先转换成精度最高的那种**,然后再进行运算
> * 细节二:当我们将**精度大的数据赋值给精度小的数据类型时,会报错**;反之会进行自动转换
> * ==细节三:byte和char,short和char这两组不会进行自动转换==
> * ==细节四:byte,short,char进行运算时,会自动转为int类型再进行，即最终是int类型，不能直接赋给更小精度的类型==
> * 细节五:boolean类型不参与转换
> * 自动提升原则:表达式结果的类型会自动提升为操作中最大的数据类型

```java
public class Hello {
    public static void main(String[] args)  {
        //细节一
        int n1 = 10;
        double d1 = n1 + 1.1;
        //float d1 = n1 + 1.1;这样写是错的,double转float,精度不够
        //float d1 = n1 + 1.1f;这样写是对的
        
        //细节二
        //int n2 = 1.1;这样写是错的
        byte b1 = 10;
        int n2 = 1;
        //byte b2 = n2;会报错,int4字节,byte1字节,不能把4字节转成1字节

        //细节三
        //char c1 = b1;会报错,byte和char不能自动转换

        //细节四
        byte b2 = 1;
        b2 = b2 + 1;  //报错,byte运算时自动转成int,int结果赋值byte类型,失败
        byte b3 = 2;
        short s1 = 1;
        s1 = s1 - 9;  //报错,int-->short,失败
        //short s2 = b2 + s1;会报错,相加后结果是int类型,int和short不能自动转换
        int n3 = b2 + s1;
        //byte b4 = b2 + b3;会报错,运算时会转成int,所以不能再赋值给byte

        //细节五
        boolean pass = true;
        //int num11 = pass;会报错,boolean类型不能转换
    }
} 
```

## 03.6 强制类型转换

> * 在将精度大的数据类型转换成精度小的数据类型时,==使用强制转换符(),但会造成精度降低或溢出==,要格外注意

```java
//示例一
public class Hello {
    public static void main(String[] args)  {
        // 将浮点数1.9转换为整数，结果为1,精度损失
        int n1 = (int)1.9;
        System.out.println("n1=" + n1);

        // 将n2强制类型转换为byte型，并赋值给b1,结果为-48,数据溢出
        int n2 = 2000;
        byte b1 = (byte)n2;
        System.out.println("b1=" + b1);
    }
} 
```

> * ==强制转换符只针对最近的操作数有效,往往会使用小括号提升优先级==

```java
public class Hello {
    public static void main(String[] args)  {
        //int x = (int)10 * 3.5 + 6 * 1.5;会报错,强转符只对10生效,表达式最终是double类型
        int x = (int)(10 * 3.5 + 6 * 1.5);  //这是对的
        System.out.println(x);
    }
```

> * ==char类型可以保存int的常量值(也要在char范围内),但不能保存int的变量值,需要强制转换==

```java
public class Hello {
    public static void main(String[] args)  {
        char c1 = 100;  //正确
        int c2 = 100;  //正确
        //char c3 = c2;  //报错,int不能和char自动转换
        char c4 = (char)c2;  //强制转换
        System.out.println(c4); //结果为d
    }
} 
```

## 03.7 基本类型和String类型转换

> * ==基本数据类型转String类型,语法:基本数据类型的值+""==
> * ==String类型转基本数据类型,语法:通过基本类型的包装类调用parseXX方法==
> * 要确保String类型可以转成支持的类型,例如"Hello"就不能转成整数

```java
public class Hello {
    public static void main(String[] args)  {
        //基本类型转String
        int n1 = 100;
        float f1 = 1.1f;
        double d1 = 1.23;
        boolean b1 = true;
        String s1 = n1 + "";
        String s2 = f1 + "";
        String s3 = d1 + "";
        String s4 = b1 + "";

        //String转基本类型
        String s5 = "123";
        int n2 = Integer.parseInt(s5);  //在oop时会详细讲
        float f2 = Float.parseFloat(s5);
        double d2 = Double.parseDouble(s5);
        boolean b2 = Boolean.parseBoolean("true");
        long l2 = Long.parseLong(s5);
        byte b3 = Byte.parseByte(s5);
        short b4 = Short.parseShort(s5);
    }
}
```

# 04 java API文档

> * 提供了Java平台中所有类和方法的详细说明，是开发者学习和使用Java技术的重要参考



