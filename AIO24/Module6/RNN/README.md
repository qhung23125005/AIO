# Recurrent Neural Networks (RNN) and Variants: LSTM & GRU
Recurrent Neural Networks (**RNNs**) are a class of neural networks designed for **sequential data processing**. Unlike traditional feedforward networks, RNNs retain information from previous time steps, making them well-suited for tasks like **natural language processing (NLP), speech recognition, and time-series forecasting**.

This document provides an overview of **RNNs** and their improved variants: **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)**.

---

## **Recurrent Neural Networks (RNN)**

### **What is an RNN?**
- RNNs process **sequential data** while maintaining a **hidden state** that captures past information.
- They **apply the same function** to each step in the sequence, making them useful for time-dependent tasks.
- Common applications include **text generation, speech recognition, and time-series forecasting**.

![image](https://github.com/user-attachments/assets/80b5c78b-1eb7-4d58-822f-ff0467d37910)


### **Limitations of RNNs**
- **Vanishing Gradient Problem:** Difficulty in learning long-term dependencies due to gradient shrinkage.
- **Short-Term Memory:** Struggles with long sequences, making it ineffective for tasks requiring deep context.
- **Limited Parallelization:** Training is slower since each time step depends on the previous step.

---

## **Long Short-Term Memory (LSTM)**

### **What is an LSTM?**
LSTMs are an improvement over RNNs, designed to **handle long-term dependencies**. They use a **cell state** and three gates:
- **Forget Gate** – Decides which past information to discard.
- **Input Gate** – Determines which new information to store.
- **Output Gate** – Controls the final output based on the cell state.

![image](https://github.com/user-attachments/assets/bc820c5c-cd7f-45be-b3ee-d763bffc3400)


### **Advantages of LSTM**
- **Solves vanishing gradient problem** by maintaining long-term dependencies.  
- **More effective for long sequences** (e.g., paragraphs, speech).  
---

## **Gated Recurrent Unit (GRU)**

### **What is a GRU?**
GRUs are a **simplified version of LSTMs**, offering similar performance with fewer parameters. They combine the **forget and input gates** into a **single "update gate"**, reducing computational cost.

![image](https://github.com/user-attachments/assets/1a69e5c6-0331-4b14-a17f-80182c31f93a)


### **Advantages of GRU**
- **Faster and more efficient than LSTM** due to fewer gates.  
- **Performs similarly to LSTM** on many tasks.  
- **Ideal for real-time applications** where speed matters.

---

## **RNN vs. LSTM vs. GRU: Key Differences**

| Feature  | RNN  | LSTM  | GRU  |
|----------|------|------|------|
| **Handles long-term dependencies?** |  No |  Yes |  Yes |
| **Computational Efficiency** |  Fast |  Slower |  Faster than LSTM |
| **Number of Gates** | 1 | 3 | 2 |
| **Suitable for Real-Time Use?** |  No |  Less Efficient |  Yes |
| **Memory Retention** |  Short-Term |  Long-Term |  Long-Term |

---

## **Exercises**
For exercise, I will apply RNN and its variance for sentiment analysis and time-series problems
