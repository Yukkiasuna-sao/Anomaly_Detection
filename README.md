# Anomaly_Detection
Try Anomaly Detection Implementation...

Doing  
1. Conceptual Understanding
    - [Outlier Analysis](https://www.springer.com/gp/book/9781461463955)

TO DO
1. Study............. 
2. Translate Anomaly Detection Kernel.......

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
