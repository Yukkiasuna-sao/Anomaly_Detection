# Anomaly_Detection
Try Anomaly Detection Implementation...

Doing 

1. Conceptual Understanding
    - [Outlier Analysis](https://www.springer.com/gp/book/9781461463955)

2. Translate Anomaly Detection Kernel.......

TO DO

1. Study.............

    1.1 [What is Anomalies? # 3](https://www.notion.so/3-1d177d5f2098402297c1b870e50f8811)
    
        - Summary & Review Report
            - Anomaly Detection Technique
                - Deep Learning Method
            - Impelementation Anomaly Detection
                - STEP1. Simple Dataset
                - STEP2. Large Dataset
                           
DONE

1.[What is Anomalies? # 1](https://www.notion.so/1-b978df92c1b34e768a3cb60d7bc2c7a4)
        
        - Summary & Review Report
            - Conceptual Definition: Anomaly Detection
            - Anomaly Detetction Characteristics

2.[What is Anomalies? # 2](https://www.notion.so/2-d5b7ec15a01846ebab2e88bebd7638ae)

        
        - Summary & Review Report
            - Anomaly Detection Technique
                - Machine Learning Method
                - Statistic Method
                
               

## What Are Anomalies?

In data mining, anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. Anomalies can be broadly categorized as:

<b>Point anomalies</b>: A single instance of data is anomalous if it's too far off from the rest. Business use case: Detecting credit card fraud based on "amount spent."

<img src="https://paper-attachments.dropbox.com/s_1185AEC62427E23657579AF288686866FF5B3F65A0E36E86D1A293C6B0CCF4B4_1553405161903_sqDCqTEGAmcjqerU4VmkGaw.png">

Source: [Introduction to Anomaly Detection in Python](https://blog.floydhub.com/introduction-to-anomaly-detection-in-python/)

<b>Contextual Anomalies</b>: The abnormality is context specific. This type of anomaly is common in time-series data. Business use case: Spending $100 on food every day during the holiday season is normal, but may be odd otherwise.

<img src="https://paper-attachments.dropbox.com/s_1185AEC62427E23657579AF288686866FF5B3F65A0E36E86D1A293C6B0CCF4B4_1554118214508_outliers4-2.png">

Source: [Introduction to Anomaly Detection in Python](https://blog.floydhub.com/introduction-to-anomaly-detection-in-python/)

<b>Collective anomalies</b>: A set of data instances collectively helps in detecting anomalies. Business use case: Someone is trying to copy data form a remote machine to a local host unexpectedly, an anomaly that would be flagged as a potential cyber attack. <b>Individual data instances may not be anomalies, but what happens together as a group.</b>

<img src="https://paper-attachments.dropbox.com/s_1185AEC62427E23657579AF288686866FF5B3F65A0E36E86D1A293C6B0CCF4B4_1553680157064_image.png">

Source: [Introduction to Anomaly Detection in Python](https://blog.floydhub.com/introduction-to-anomaly-detection-in-python/)

Anomaly detection is similar to — but not entirely the same as — <b>noise removal</b> and <b>novelty detection</b>.

<b>Novelty detection</b> is concerned with identifying an unobserved pattern in new observations not included in training data like a sudden interest in a new channel on YouTube during Christmas, for instance.

<b>Noise removal (NR)</b> is the process of removing noise from an otherwise meaningful signal.

## Reference

- [1] Liu, Fei Tony; Ting, Kai Ming; Zhou, Zhi-Hua (December 2008). "Isolation Forest". 2008 Eighth IEEE International Conference on Data Mining: 413–422. [[Paper]](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)
