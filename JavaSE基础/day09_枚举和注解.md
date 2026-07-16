# 01 枚举

> * 枚举（`enum`）是一种特殊的数据类型，用于定义一组常量。枚举类型提供了一种类型安全的方式来表示一组固定的常量值
> * ==自己理解：枚举就是个类，是已经定义好确定对象的类，不能除此之外再创建更多的对象==
> * 因而枚举类也可以添加构造器、字段、方法

## 01.1 自定义类实现枚举

> * 在Java 5之前，没有`enum`关键字，枚举通常是通过自定义类来实现的。这种方式依赖于类的静态字段和构造函数来定义和管理枚举常量

> * #### 自定义类实现枚举的步骤
>
>   1. **定义类**：定义一个类，并将构造函数私有化
>   2. **定义常量**：在类中定义静态常量(通常名字大写)，代表每个枚举值
>   3. **私有字段**：定义私有字段来保存枚举常量的属性
>   4. **构造函数**：定义一个私有构造函数来初始化这些属性
>   5. **访问方法**：提供公共方法(get)来访问这些属性，不能设置set方法

```java
class Day {
    // 定义枚举常量
    public static final Day SUNDAY = new Day("Sunday");
    public static final Day MONDAY = new Day("Monday");
    public static final Day TUESDAY = new Day("Tuesday");
    public static final Day WEDNESDAY = new Day("Wednesday");
    public static final Day THURSDAY = new Day("Thursday");
    public static final Day FRIDAY = new Day("Friday");
    public static final Day SATURDAY = new Day("Saturday");
    //私有字段
    private String name;

    // 私有构造函数
    private Day(String name) {
        this.name = name;
    }

    // 访问方法
    public String getName() {
        return name;
    }

    // 静态方法返回所有枚举值
    public static Day[] values() {
        return new Day[]{SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY};
    }

    @Override
    //toString方法是Object类的一个方法，当输出对象时，默认输出的是toString()方法的返回值
    public String toString() {
        return name;
    }
}

public class Main {
    public static void main(String[] args) {
        // 使用自定义类枚举
        Day today = Day.MONDAY;
        System.out.println("Today is: " + today);

        // 打印所有枚举值
        for (Day day : Day.values()) {  //
            System.out.println(day);
        }
    }
}
```

## 01.2 enum关键字实现枚举

> * Java 5 引入了 `enum` 关键字，使得定义枚举更加简洁和类型安全。`enum` 提供了许多内置功能，如内置的 `values()` 方法、枚举常量的序数、枚举常量的名称等
> * ==要求把常量对象的定义写在最前面==
> * 使用`enum`关键字创建枚举类时，默认会继承`Enum`类（可通过反编译.class查看）
> * ==枚举对象之间用逗号间隔开，最后一个对象后用分号==
> * 如果是无参构造器，则枚举对象的参数列表和小括号都可以省略

```java
enum Day {
    SUNDAY("Sunday"),   //就是常量名带着构造器的参数列表
    MONDAY("Monday"),
    TUESDAY("Tuesday"),
    WEDNESDAY("Wednesday"),
    THURSDAY("Thursday"),
    FRIDAY("Friday"),
    SATURDAY("Saturday");

    private String name;

    // 枚举的构造函数
    private Day(String name) {
        this.name = name;
    }

    // 访问方法
    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return name;
    }
}

public class TestEnumDay {
    public static void main(String[] args) {
        // 使用枚举
        Day today = Day.MONDAY;
        System.out.println("Today is: " + today);

        // 打印所有枚举值
        for (Day day : Day.values()) {
            System.out.println(day);
        }

        // 使用内置方法
        Day day = Day.valueOf("FRIDAY");
        System.out.println("Day: " + day);
        System.out.println("Ordinal of " + day + ": " + day.ordinal());
        System.out.println("Name of the enum constant: " + day.name());
    }
}
```

|   特性   | 自定义类实现枚举                       | enum实现枚举                                           |
| :------: | -------------------------------------- | ------------------------------------------------------ |
| 定义方式 | 通过类和静态常量实现                   | 使用 `enum` 关键字定义                                 |
| 类型安全 | 不够类型安全，依赖于静态常量的正确使用 | 类型安全，编译器保证类型正确性                         |
| 内置方法 | 需要手动定义 `values()` 方法           | 提供内置的 `values()`、`valueOf()`、`ordinal()` 等方法 |
| 序数 | 需要手动实现序数功能                   | 提供内置的 `ordinal()` 方法                            |
|  可读性  | 代码较为冗长，容易出错                 | 代码简洁，可读性强，减少错误                           |
| 功能扩展 | 通过静态方法和字段扩展功能             | 通过方法、字段和构造函数扩展功能                       |

## 01.3 enum内置方法

### 01.3.1 `value`方法

> * `values()` 方法返回一个包含枚举类型中所有枚举常量的数组。这个方法在枚举类型中自动生成，允许我们遍历所有的枚举常量                                                                                                                           

```java
enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;
}

public class Main {
    public static void main(String[] args) {
        Day[] days = Day.values();
        for (int i = 0; i < days.length; i++) {
            System.out.print(days[i] + " ");
        }
    }
}
```

### 01.3.2 `valueOf`方法

> * `valueOf(String name)` 方法返回具有指定名称的枚举常量,即返回枚举类中指定名称的实例。如果没有具有指定名称的常量，则抛出 `IllegalArgumentException`

```java
enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;
}

public class Main {
    public static void main(String[] args) {
        Day day = Day.valueOf("FRIDAY");
        System.out.println("Day: " + day);  //会调用toString方法
    }
}
```

### 01.3.3 `ordinal`方法

> * `ordinal()` 方法返回枚举常量在枚举类型中的序数（从0开始）

```java
enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;
}

public class Main {
    public static void main(String[] args) {
        Day day = Day.valueOf("FRIDAY");  //获取一个实例对象
        //或者Day day = Day.MONDAY;
        System.out.println("Ordinal of " + day + ": " + day.ordinal());
    }
}
```

### 01.3.4 `name`方法

> * `name()` 方法返回枚举常量的名称，即定义枚举常量时使用的标识符

```java
enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;
}

public class Main {
    public static void main(String[] args) {
        Day day = Day.MONDAY;
        System.out.println("Name of the enum constant: " + day.name());
    }
}
```

### 01.3.5 `compareTo`方法

> * `compareTo(比较对象)` 方法比较两个枚举常量的序数。实现了 `Comparable` 接口
> * `compareTo`方法的返回值是一个整数，表示两个对象的比较结果：
>   如果两个对象相等，返回0
>   如果当前对象小于比较对象，返回负整数
>   如果当前对象大于比较对象，返回正整数

```java
enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;
}

public class Main {
    public static void main(String[] args) {
        Day day1 = Day.MONDAY;  
        Day day2 = Day.FRIDAY;

        int comparison = day1.compareTo(day2);
        if (comparison < 0) {
            System.out.println(day1 + " comes before " + day2);
        } else if (comparison > 0) {
            System.out.println(day1 + " comes after " + day2);
        } else {
            System.out.println(day1 + " is the same as " + day2);
        }
    }
}
```

### 01.3.6 `toString`方法

> * `toString()` 方法返回枚举常量的名称，通常与 `name()` 方法的返回值相同。可以重写 `toString()` 方法以提供更友好的字符串表示

```java
enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;

    @Override
    public String toString() {
        // 自定义字符串表示
        return "Day: " + name();
    }
}

public class Main {
    public static void main(String[] args) {
        Day day = Day.MONDAY;
        System.out.println(day.toString());
    }
}
```

## 01.4 枚举类实现接口

> * 枚举类型不仅可以包含字段、构造函数和方法，还可以实现接口。这使得枚举类型更加灵活和强大，可以用于更复杂的应用场景
>
> * #### 语法
>
>   1. **定义接口**：定义一个接口，其中包含一个或多个抽象方法。
>   2. **定义枚举**：使用`enum`关键字定义枚举，并实现接口。
>   3. **实现接口方法**：为每个枚举常量提供接口方法的具体实现

```java
//定义接口
interface Operation {
    double apply(double x, double y);
}

//定义枚举并实现接口
enum OperationType implements Operation {
    ADDITION {
        @Override
        public double apply(double x, double y) {
            return x + y;
        }
    },
    SUBTRACTION {
        @Override
        public double apply(double x, double y) {
            return x - y;
        }
    },
    MULTIPLICATION {
        @Override
        public double apply(double x, double y) {
            return x * y;
        }
    },
    DIVISION {
        @Override
        public double apply(double x, double y) {
            if (y == 0) {
                throw new IllegalArgumentException("Division by zero");
            }
            return x / y;
        }
    }
}

//使用枚举
public class Main {
    public static void main(String[] args) {
        double a = 10.0;
        double b = 5.0;

        // 使用枚举类型执行运算
        for (OperationType operation : OperationType.values()) {
            System.out.printf("%f %s %f = %f%n", a, operation, b, operation.apply(a, b));
        }
    }
}
```

# 02 注解

> * 注解（Annotations）是Java提供的一种元数据机制，可以为代码元素（类、方法、变量、参数等）提供附加信息。注解不会直接影响程序的逻辑，但可以通过工具或框架在编译、类加载、运行时进行处理和利用
> * 在javase中注解的使用较为简单，但在javaee中，注解占据了重要的角色

## 02.1 内置基本注解

### 02.1.1 @override

> * 重写方法的注解，是可选的，不写也可以

```java
class A{
    public  void fun1(){
        System.out.println("父类A方法");
    }
}

class B extends A{
    @Override
    public void fun1(){
        System.out.println("子类B方法");
    }
}
```

### 02.1.2 @deprecated

> * 用于标识过时的方法、类或字段
> * 也就是不再推荐使用，但用还是可以用

### 02.1.3 @suppresswarnings

> * 抑制编译器警告

## 02.2 JDK元注解

> * ==元注解是用于修饰其他注解的注解==。Java提供了一些内置的元注解，用于定义自定义注解的行为和约束。常见的元注解包括：
>   1. **`@Retention`**：指定注解的保留策略
>   2. **`@Target`**：指定注解可以应用的目标元素
>   3. **`@Documented`**：指定注解是否包含在Javadoc中
>   4. **`@Inherited`**：指定注解是否可以被子类继承
>   5. **`@Repeatable`**：指定注解是否可以重复使用
