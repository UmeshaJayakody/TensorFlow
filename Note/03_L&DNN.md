## 1. Linear Model

### What it is

A **linear model** works like a **calculator**.

It:

* Looks at each thing **one by one**
* Gives each thing a **fixed score**
* Adds all scores
* Decides survive / not survive

---

### Simple example (Titanic)

Assume the model learned these rules:

| Feature   | Rule      |
| --------- | --------- |
| Male      | −2 points |
| Female    | +2 points |
| 3rd class | −2 points |
| 1st class | +2 points |
| Young     | +1 point  |
| Old       | −1 point  |

---

### Passenger A

**Male, 3rd class, young**

Calculation:

```
Male        → -2
3rd class  → -2
Young      → +1
----------------
Total      → -3  (Low chance)
```

Model says **likely died**

---

### Passenger B

**Male, 1st class, young**

```
Male        → -2
1st class  → +2
Young      → +1
----------------
Total      → +1  (Medium chance)
```

Model says **maybe survived**

---

### Problem with linear model

The rule **"Male" is always −2**
The rule **"3rd class" is always −2**

The model **cannot understand**:

> "Male AND 3rd class together is much worse than normal"

It only **adds numbers**, nothing more.

---

## 2. Neural Network — explained simply

### What it is

A **neural network** works like a **thinking brain**.

It:

* Looks at features **together**
* Learns **hidden rules**
* Changes decisions based on combinations

---

### Same Titanic example

Instead of fixed rules, it learns patterns like:

* Male + 3rd class → **VERY low survival**
* Male + 1st class → **OK survival**
* Female + child → **VERY high survival**

---

### Passenger A (Male, 3rd class)

Neural network thinks:

> "This combination happened many times → most died"

**Very low survival**

---

### Passenger B (Male, 1st class)

Neural network thinks:

> "This combination often survived"

**High survival**

---

## 3. Why linear fails but neural works (key idea)

### Linear model

* Treats features **separately**
* Same rule always
* Cannot learn **IF–AND logic**

### Neural network

* Combines features
* Learns **IF this AND that**
* Learns **curves and complex patterns**

---

## 4. Very simple analogy (best to remember)

### Linear model

**Calculator**

```
Add scores → decide
```

### Neural network

**Brain**

```
Recognize patterns → decide
```

---

## 5. One-line summary (exam-ready)

* **Linear model**: simple, fast, explainable, but weak for complex data
* **Neural network**: powerful, learns combinations, but harder to explain
