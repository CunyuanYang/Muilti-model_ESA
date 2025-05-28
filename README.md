![image](https://github.com/user-attachments/assets/531c5ca2-e37e-4fb3-a9ce-fe65b1762e27)Traditional sentiment analysis methods primarily focus on single-modality data, such as analyzing text or images alone. However, single modality approaches often fail to capture the complexity of users' emotions comprehensively.[11] For instance, while text may convey a certain sentiment, accompanying images or audio might convey subtler or even contradictory emotional information. Therefore, combining multimodal data for sentiment analysis is crucial, as it can enhance the accuracy and robustness of sentiment detection.

The four innovations of this study can be summarized as follows:
1. Integration of BERT and ResNet for Multimodal Graph Construction: This study uniquely combines BERT for text encoding and ResNet for image feature extraction to construct separate text and image graphs.
2. Application of Multi-Head Attention Mechanism for Multimodal Fusion: The study introduces a multi-head attention mechanism to weight and integrate the nodes and edges of the text graph and image graph. 
3. Dual Loss Functions for Balanced Optimization: The study combines two distinct loss functions—one for maximizing sentiment prediction accuracy and another for ensuring class balance.
4. Introduces GraphSAGE as an innovative approach, leveraging its efficient neighbor sampling and feature aggregation mechanism to address the computational bottlenecks of traditional graph convolutional networks when handling large-scale graph data.

This comprehensive architecture effectively captures and integrates multimodal information, leveraging the strengths of BERT, ResNet, multi-head attention, and GraphSAGE to deliver robust sentiment analysis. The structure of the proposed model is shown below：
![image](https://github.com/user-attachments/assets/105cf018-7f70-4fd5-ae2c-95e97b782d17)
