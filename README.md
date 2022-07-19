# FAtNet: Cost-Effective Approach Towards Mitigating the Linguistic Bias in Speaker Verification Systems

This repository contains the implementation of our paper : _FAtNet: Cost-Effective Approach Towards Mitigating the Linguistic Bias in Speaker Verification Systems_

This work has been accepted at the **Findings of NAACL'22**

Link to the paper: https://aclanthology.org/2022.findings-naacl.93/


_**If you use this repository, please cite the following paper:**_

**Divya Sharma and Arun Balaji Buduru. 2022. FAtNet: Cost-Effective Approach Towards Mitigating the Linguistic Bias in Speaker Verification Systems. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1247–1258, Seattle, United States. Association for Computational Linguistics.**


@inproceedings{sharma-buduru-2022-fatnet,

    title = "{FA}t{N}et: Cost-Effective Approach Towards Mitigating the Linguistic Bias in Speaker Verification Systems",
    
    author = "Sharma, Divya  and
      Buduru, Arun Balaji",
      
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    
    month = jul,
    
    year = "2022",
    
    address = "Seattle, United States",
    
    publisher = "Association for Computational Linguistics",
    
    url = "https://aclanthology.org/2022.findings-naacl.93",
    
    pages = "1247--1258",
    
    abstract = "Linguistic bias in Deep Neural Network (DNN) based Natural Language Processing (NLP) systems is a critical problem that needs attention. The problem further intensifies in the case of security systems, such as speaker verification, where fairness is essential. Speaker verification systems are intelligent systems that determine if two speech recordings belong to the same speaker. Such human-oriented security systems should be usable by diverse people speaking varied languages. Thus, a speaker verification system trained on speech in one language should generalize when tested for other languages. However, DNN-based models are often language-dependent. Previous works explore domain adaptation to fine-tune the pre-trained model for out-of-domain languages. Fine-tuning the model individually for each existing language is expensive. Hence, it limits the usability of the system. This paper proposes the cost-effective idea of integrating a lightweight embedding with existing speaker verification systems to mitigate linguistic bias without adaptation. This work is motivated by the theoretical hypothesis that attentive-frames could help generate language-agnostic embeddings. For scientific validation of this hypothesis, we propose two frame-attentive networks and investigate the effect of their integration with baselines for twelve languages. Empirical results suggest that frame-attentive embedding can cost-effectively reduce linguistic bias and enhance the usability of baselines.",

}
