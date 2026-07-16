# 01 顺序控制

> * 程序从上到下逐行执行,中间没有任何判断和跳转
> * **定义成员变量时采用合理的前向引用(也就是先声明及定义,再使用,不能反)**

# 02 分支控制if-else

> * 有单分支,双分支,多分支三种情况
> * **单分支、多分支可以没有else**
## 02.1 单分支

```java
import java.util.Scanner;
public class If01 {  
    public static void main(String[] args) { 
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入年龄：");
        int age = sc.nextInt();

        //单分支,就一个if
        if(age >= 18){  //判断条件为真时,执行下面大括号里的代码;为假时,则略过该代码
            System.out.println("你已成年");
        }
    }  
}
```
## 02.2 双分支

```java
import java.util.Scanner;      
public class If01 {  
    public static void main(String[] args) { 
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入年龄：");
        int age = sc.nextInt();

        //双分支,一个if,一个else
        if(age >= 18){  //当此处的判断条件为假时,则执行esls后{}内的代码
            System.out.println("你已成年");
        }
        else{
            System.out.println("还未成年");
        }
    }  
}
```
## 02.3 多分支

```java
import java.util.Scanner;
public class If01 {  
    public static void main(String[] args) { 
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入芝麻信用分数：");
        int score = sc.nextInt();
        if(score == 100){
            System.out.println("信用极好");
        }else if(score >= 80){
            System.out.println("信用优秀");
        }else if(score >= 60){
            System.out.println("信用一般");
        }else{
            System.out.println("信用不及格");
        }    
    }
}
```

## 02.4 嵌套分支

```java
import java.util.Scanner;
public class If01 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.println("请输入芝麻信用分数：");
        int score = sc.nextInt();
        System.out.println("是否为新用户？(是/否)");
        String isNewUser = sc.next();
        
        //equalsIgnoreCase() 是 Java 中 String 类的一个方法，用于比较两个字符串是否相等，忽略大小写
        if (isNewUser.equalsIgnoreCase("是")) {
            System.out.println("欢迎新用户！");
            if (score >= 0 && score < 60) {
                System.out.println("作为新用户，您的信用评分为 " + score + "，建议保持良好的信用行为。");
            }
			else {
                // 这里直接使用外部的逻辑判断，避免重复，但为了展示嵌套，我们保持结构
                if (score == 100) {
                    System.out.println("信用极好");
                }
                else if (score >= 80) {
                    System.out.println("信用优秀");
                }
                else if (score >= 60) {
                    System.out.println("信用一般");
                }
                else {
                    System.out.println("信用不及格，请注意信用积累。");
                }
            }
        }
        else {
            // 老用户的处理逻辑可以在这里添加，为了简洁，我们沿用之前的逻辑
            if (score == 100) {
                System.out.println("信用极好");
            }
            else if (score >= 80) {
                System.out.println("信用优秀");
            }
            else if (score >= 60) {
                System.out.println("信用一般");
            }
            else {
                System.out.println("信用不及格");
            }
        }
    }
}
```

```java
import java.util.Scanner;
public class If01 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入您的成绩：");
        double score = sc.nextDouble();

        if(score > 8.0){
            System.out.println("请输入您的性别(男或女)：");
            String gender = sc.next();
            if ("男".equals(gender)) {
                System.out.println("恭喜你进入男子组决赛");
            } else {
                System.out.println("恭喜你进入女子组决赛");
            }
        }else{
            System.out.println("您未进入决赛");
        }
    }
}
```

## 02.5 if练习

```java
//练习一
public class If01 {
    public static void main(String[] args){
        int x = 7;
        int y = 4;
        if(x > 5){
            if(y > 5){
                System.out.println(x + y);  //所属的if层判断条件为false,不输出
            }
            System.out.println("hello");  //所属是最外层if,判断条件为true,会输出
        }
        else{
            System.out.println( "x = " + x);  //对应if为true,所以这个else不输出
        }
    }
}
```

```java
//练习二
public class If01 {
    public static void main(String[] args){
        double a = 32.4;
        double b = 432.63;
        if(a > 10.0 && b < 20.0){
            System.out.println(a + b);
        }
        else{
            System.out.println("程序结束");
        }
    }
}
```

```java
//练习三
public class If01 {
    public static void main(String[] args){
        int a = 7;
        int b = 8;
        if((a + b) % 3 == 0 && (a + b) % 5 == 0){
            System.out.println("两数之和满足既被3又被5整除");
        }
        else{
            System.out.println("两数之和不满足既被3又被5整除");
        }
    }
}
```

```java
//练习四
import java.util.Scanner;
public class If01 {  
    public static void main(String[] args) { 
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入年份：");
        int year = sc.nextInt();
        if((year % 4 ==0 && year % 100 != 0) || (year % 400 == 0)){
            System.out.println("这个年份是闰年");
        }
        else{
            System.out.println("这个年份不是闰年");
        }
    }  
}
```

## 02.6 switch分支结构

> * 表达式对应一个值
> * 当表达式的值等于常量1,就执行语句块1;不匹配则继续匹配常量2
> * 如果一个都没匹配上,则执行default

 ```java
 //基本语法
 switch(表达式){
         case 常量1：
         	语句块1;
         	break;  //可选,跳出switch结构,如果没有break,则接下来会执行语句块2
         case 常量2:
         	语句块2;
         	break;  //可选,如果没有,则接下来直接执行语句块3
         ...
         default:  //可选
         	语句块;
         	break
 }
 ```

```java
import java.util.Scanner;
public class Switch01 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入一个字符(a~e)：");
        char inputchar = sc.next().charAt(0);  // 读取用户输入的第一个字符

        switch(inputchar){
            case 'a':
                System.out.println("星期一");
                break;
            case 'b':
                System.out.println("星期二");
                break;
            case 'c':
                System.out.println("星期三");
                break;
            case 'd':
                System.out.println("星期四");
                break;
            case 'e':
                System.out.println("星期五");
            default:
                System.out.println("输入错误");
        }
        System.out.println("退出switch,继续执行接下来的代码");
    }
}
```

> * 表达式必须是一个产生单个值的表达式，==其类型应是`byte`、`short`、`char`、`int`、`String`（从Java 7开始）、或者是枚举（`enum`）类型==
> * **表达式返回值的数据类型应该和case后的常量的类型一致,或是可以自动转成可以相互比较的类型**,也就是表达式可以是`byte`、`short`、`char`,而case常量是`int`,因为java会自动把前三种类型转成`int`
> * **case子句中的值必须是常量(诸如：1,'d')或常量表达式,不能是变量**

```java
byte b = 1;  
switch (b) { // b被自动提升为int  
    case 1:  
        System.out.println("One (byte)");  
        break;  
    // ...  
} 
  
short s = 1;  
switch (s) { // s被自动提升为int  
    case 1:  
        System.out.println("One (short)");  
        break;  
    // ...  
}  
  
char ch = 'A';  
switch (ch) { // char在比较时被视为其ASCII值的int  
    case 'A': // 实际上是与int类型的65进行比较  
        System.out.println("A (char)");  
        break;  
    // ...  
}
```

## 02.7 swtich练习

```java
import java.util.Scanner;
public class Switch01 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入你的成绩：");
        // 读取用户输入的第一个字符
        double score = sc.nextDouble();
        // 计算成绩是否合格，这里将成绩除以60，取整数部分，用于后续的判断
        // 大于60结果为1,小于60结果为0,做到了二分
        int temp = (int)(score / 60);

        switch(temp){
        //switch ((int)(score / 60)){
            case 1:
                System.out.println("成绩大于60,合格");
                break;
            case 0:
                System.out.println("成绩小于60,不合格");
                break;
            default:
                System.out.println("输入有误");
        }
        System.out.println("退出switch,继续执行接下来的代码");
    }
}
```

```java
import java.util.Scanner;
public class Switch01 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入月份(1~12)：");
        // 读取用户输入的第一个字符
        int month = sc.nextInt();

        //用穿透效果
        switch(month){
            case 3:
            case 4:
            case 5:
                System.out.println("春季");
                break;
            case 6:
            case 7:
            case 8:
                System.out.println("夏季");
                break;
            case 9:
            case 10:
            case 11:
                System.out.println("秋季");
                break;
            case 12:
            case 1:
            case 2:
                System.out.println("冬季");
                break;
            default:
                System.out.println("输入有误");
        }
        System.out.println("退出switch,继续执行接下来的代码");
    }
}
```

## 02.8 if和switch比较

> * 如果判断的具体数值不多,符合byte,short,int,char,enum,String这6种类型,建议用switch
> * 其他情况:对区间判断,对结果为boolean类型判断可使用if,if适用范围更广,也更容易写

# 03 循环

## 03.1 for循环

> * 当语句块只有一条语句时,可以省略{},不过最好还是别,方便清晰阅读
> * **循环条件是返回一个boolean类型的表达式**
> * ==for循环可看做是while循环的一种简化形式==

```java
//基本语法
for(循环变量初始化;循环条件;循环变量迭代){
    语句块;
}
```

```java
public class Switch01 {
    public static void main(String[] args) {
        for(int i = 0; i < 10; i++){
            System.out.println("这是for循环" + i);
        }
    }
}
```

> * 循环变量初始化和循环条件可以写在外面,但分号不能省略
> * 写在外面,使得for循环结束后,还能使用该变量,相当于扩大了其使用范围
```java
int i = 1;
for( ; i <= 10; ){
    System.out.println("这是for循环" + i);
    i++
}
System.out.println(i);  //i = 11
```

```java
//表示无限循环,死循环
for(;;){
    System.out.println("这是无限循环");
}
```

> * **循环初始值可以有多条初始化语句,但要求类型一致,并且中间用逗号隔开**
> * **循环变量迭代也可以有多条迭代语句,中间用逗号隔开**

```java
public class Switch01 {
    public static void main(String[] args) {
        for(int i = 0, j = 0; i < 10; i++, j += 2){
            System.out.println("i = " + i + " j = " + j);
        }
    }
}
```

## 03.2 for循环练习

```java
//统计100以内所有是9的倍数的整数,统计个数及总和
public class Switch01 {
    public static void main(String[] args) {
        int count = 0;
        int sum = 0;
        for(int i = 1; i < 101; i++){
            if(i % 9 == 0){
                count++;
                sum += i;
                System.out.println("i = " + i);
            }
        }
        System.out.println("count = " + count);  // 11个
        System.out.println("sum = " + sum);  //594
    }
}
```

```java
//打印一列等式,第一个加数为0~5,第二个加数为5~0
public class Switch01 {
    public static void main(String[] args) {
        for(int i = 0,j = 5; i < 6; i++,j--){
            System.out.println(i + "+" + j + "=" + (i+j));
        }
    }
}
```

## 03.3 while循环

> * while循环同样也是四要素,只是位置和for循环不一样
> * 先判断再执行

```java
//基本语法
循环变量初始化;
while(循环条件){
    语句块;
    循环变量迭代;
}
```

```java
public class While01 {
    public static void main(String[] args) {
        int i = 1;
        while(i <= 10){
            System.out.println("hello world" + i);
            i++;
        }
    }
}
```

## 03.4 while循环练习

```java
//打印100以内能被3整除的数
public class While01 {
    public static void main(String[] args) {
        int i = 1;
        while(i <= 100){
            if(i % 3 ==0){
                System.out.println("i=" + i); 
            }
            i++;
        }
    }
}
```

## 03.5 do..while循环

> * **先执行再判断,也就是说一定会至少执行一次**
> * 最后有一个分号

```java
//基本语法
循环变量初始化;
do{
    语句块;
    循环变量迭代;
}while(循环条件);
```

```java
public class Dowhile01 {
    public static void main(String[] args) {
        int i = 1;
        do{
            System.out.println("helloworld" + i);
            i++;
        }while(i <= 10);
    }
}
```

```java
//计算1~100的和
public class While01 {
    public static void main(String[] args) {
        int i = 1;
        int sum = 0;
        do{
            sum += i;
            i++;
        }while(i <= 100);
        System.out.println(sum);  //5050
    }
}
```

```java
//统计1~200之间能被5整除但不能被3整除的数的个数
public class While01 {
    public static void main(String[] args) {
        int i = 1;
        int count = 0;
        do{
            if(i % 5 == 0 && i % 3 != 0){
                count ++;
            }
            i++;
        }while(i <= 200);
        System.out.println(count);  //27个
    }
}
```

```java
//不还钱,就一直打他,直到说还钱
public class While01 {
    public static void main(String[] args) {
        char answer = ' ';
        int i = 1;
        do{
            System.out.println("piapiapia~");
            Scanner sc = new Scanner(System.in);
            System.out.println("你还钱么?请输入选择(y/n)：");
            answer = sc.next().charAt(0);
            i++;
        }while(answer == 'n');
        System.out.println("那就不打你了");
    }
}
```

## 03.6 多重循环

> * for,while,do...while均可作为外层或是内层循环,建议不要超过三层循环,不然代码可读性差
> * 当内层循环的循环条件为false时,才可结束外层的当次循环,开始外层的下一次循环
> * 设外层循环次数为`n`次,内层循环次数为`m`次,则一共循环`n*m`次

```java
public class MultiplyFor{
    public static void main(String[] args) {
        for(int i = 0;i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                System.out.print(i + "*" + j + "=" + i*j + "\t");
            }
        }
    }
}
```

```java
//统计3个班(每班5人)的班平均分,以及三个班的平均分,还有总的及格人数
import java.util.Scanner;
public class MultiplyFor{
    public static void main(String[] args) {
        int pass_num = 0;
        double all_score = 0;
        double aver_all_score = 0;
        for(int i = 1; i <= 3; i++){
            double class_score = 0;
            double aver_class_score = 0;
            for(int j = 1; j <= 5; j++){
                System.out.println("请输入第" + i + "班的第" 
                + j + "个学生的成绩:");
                Scanner input = new Scanner(System.in);
                double score = input.nextDouble();
                class_score += score;
                if(score >= 60){
                    pass_num++;
                }
            }
            aver_class_score = class_score / 5;
            System.out.println("第" + i +"个班的均分为" + aver_class_score);
            all_score += class_score;
        }
        aver_all_score = all_score / 15;
        System.out.println("所有班级的平均分为" + aver_all_score);
        System.out.println("及格人数为" + pass_num);
    }
}
```

```java
//九九乘法口诀表
public class MultiplyFor{
    public static void main(String[] args) {
        for(int i = 1; i < 10; i++) {
            for(int j = 1; j <= i; j++) {
                System.out.print(j + "*" + i + "=" + i*j + "\t");
            }
            System.out.println();  //换行
        }
    }
}
```

```java
//打印空心金字塔
//P136
```

## 03.7 增强for循环

> * 也称为 "for-each" 循环，是 Java 中引入的一种更简洁、易读的循环结构，用于遍历数组和集合等可迭代对象。增强 `for` 循环在遍历集合或数组时，不需要显式地处理索引或迭代器，大大简化了代码的书写和阅读

```java
//基本语法
for (elementType element : collectionOrArray) {
    // 使用 element 执行一些操作
}
```

> * **`elementType`**: 迭代的元素的类型（与数组或集合中存储的类型相匹配）
> * **`element`**: 当前迭代的元素，在每次循环中它将持有集合或数组中的一个元素
> * **`collectionOrArray`**: 要遍历的集合或数组

> * 增强 `for` 循环非常适合遍历数组，因为它不需要显式地处理数组的索引

```java
public class EnhancedForLoopArray {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};

        // 使用增强 for 循环遍历数组
        for (int number : numbers) {
            System.out.println("Number: " + number);
        }
    }
}
```

```java
import java.util.ArrayList;
import java.util.List;

public class EnhancedForLoopCollection {
    public static void main(String[] args) {
        List<String> fruits = new ArrayList<>();
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");

        // 使用增强 for 循环遍历集合
        for (String fruit : fruits) {
            System.out.println("Fruit: " + fruit);
        }
    }
}
```

> * 增强 `for` 循环主要解决了传统 `for` 循环在遍历数组和集合时的繁琐操作，尤其是对于需要频繁访问集合中的元素时
> * 增强 `for` 循环更简洁，更不容易出错，因为它避免了索引越界等常见问题

> * **无法修改集合元素**: 增强 `for` 循环不提供访问元素索引的能力，也无法直接修改集合或数组中的元素。如果需要修改元素的值，仍需使用传统的 `for` 循环或迭代器
> * **无法移除集合中的元素**: 如果你需要在遍历集合时删除元素，增强 `for` 循环无法完成此操作。在这种情况下，你应该使用 `Iterator`
> * **适用于实现了 `Iterable` 接口的集合**: 增强 `for` 循环可以用来遍历所有实现了 `Iterable` 接口的集合（如 `List`, `Set` 等），以及数组

# 04 break

> * break语句用于终止某个语句块的执行,常用在switch和循环中

```java
public class Break01{
    public static void main(String[] args) {
        for(int i = 0; i < 10; i++){
            if(i ==3){
                break;
            }
            System.out.println("ok" + i);
        }
    }
}
```

> * ==当break语句出现在多层嵌套时,可以通过标签指明要终止的是哪一层语句块==
> * ==标签名字可以自由命名，符合标识符命名规则即可，必须紧跟一个冒号==
> * ==break后指定到哪个label，程序就会跳转到贷标签的语句块的末尾==，若没有指定，则就近退出
> * 在实际开发中，尽量不适用标签

```java
public class Break01{
    public static void main(String[] args) {
        label1:
        for(int j = 0; j < 4; j++) {
        label2:
            for(int i = 0; i < 10; i++) {
                if(i == 2){
                    break label1;  //如果没有标签,这个break会结束内层for循环
                }
                System.out.println("i = " + i);  //最终输出i=0 i=1
                //如果没有标签,则会输出4对i=0 i=1
            }
        }   
    }
}
```

```java
//求1~100以内整数的和,当和第一次大于20时的那个数
public class Break01{
    public static void main(String[] args) {
        int sum = 0;
        label1:
        for(int j = 1; j < 101; j++) {
            sum += j;
            if(sum > 20) {
                System.out.println(j);
                break label1;
            }
        }    
    }
}
```

```java
import java.util.Scanner;
public class Break01{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);   

        for(int i = 0; i < 3; i++){
            System.out.println("请输入用户名：");
            String name = sc.next();
            System.out.println("请输入密码：");
            String keyword = sc.next();
            if(name.equals("丁真") && keyword.equals("666")){
                System.out.println("恭喜你，登录成功！");
                break;
            }else{
                System.out.println("您还剩" + (2 - i) + "次机会！");
            }
        }
    }
}
```

# 05 continue

> * 用于结束本次循环,继续执行下一次循环(相当于是轮空的意思)
> * 在出现多层嵌套时,也可以使用标签来指明是跳过哪层循环

```java
public class Continue01{
    public static void main(String[] args) {
        int i = 1;
        while(i <= 4){
            i++;
            if(i == 2){
                continue;
            }
            System.out.println("i = " + i);  //结果为:3 4 5
        }
    }
}
```

```java
public class Continue01{
    public static void main(String[] args) {
        label1:
        for(int j = 1; j <= 4; j++){
            label2:
            for(int i = 0;i < 4; i++){
                if(i ==2){
                    continue;  //结果为013013013013
                    //continue label1;//轮空此次外层循环,结果为01010101
                }
                System.out.println("i = " + i);
            }
        }
    }
} 
```

# 06 return

> * 若在方法里使用return,表示跳出所在的方法.如果写在main方法,则会退出程序

```java
public class Return01{
    //编写一个main方法
    public static void main(String[] args) {
        for(int i = 1;i <= 5;i++){
            if(i == 3){
                System.out.println("ok");
                return;  //return用在main方法,结束整个程序
            }
            System.out.println("i=" + i);
        }
        System.out.println("end");
    }
}  //最后输出i=1,i=2,ok
```

# 07 综合练习

```java
public class Exercise{
    public static void main(String[] args) {
        double money = 100000;
        int count = 0;
        int i = 0;
        while(i >= 0){
            if(money > 50000){
                money -= money*0.05;
                count += 1;
            }else if(1000 <= money && money < 50000){
                money -= 1000;
                count += 1;
            }else{
                break;
            }
            i++;
        }
        System.out.println(count);  //62次
    }
} 

```

