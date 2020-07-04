# Learning to Cluster under Domain Shift

<p align="center">
    <img src="./imgs/teaser2.png"/> <br />
    <em> 
    Figure 1. Illustration of the proposed Unsupervised Clustering under Domain Shift setting.
    </em>
</p>

> **Learning to Cluster under Domain Shift**<br>
> [Willi Menapace](https://github.com/willi-menapace), [Stéphane Lathuilière](https://github.com/Stephlat/), [Elisa Ricci](http://elisaricci.eu/)<br>
> ECCV 2020<br>

> Paper: [link available soon]()<br>

> **Abstract:** *While  unsupervised  domain  adaptation  methods  based  ondeep architectures have achieved remarkable success in many computervision tasks, they rely on a strong assumption, i.e. labeled source datamust be available. In this work we overcome this assumption and we ad-dress the problem of transferring knowledge from a source to a target do-main when both source and target data have no annotations. Inspired byrecent works on deep clustering, our approach leverages information fromdata gathered from multiple source domains to build a domain-agnosticclustering model which is then refined at inference time when target databecome available. Specifically, at training time we propose to optimize anovel information-theoretic loss which, coupled with domain-alignmentlayers, ensures that our model learns to correctly discover semantic labelswhile discarding domain-specific features. Importantly, our architecturedesign ensures that at inference time the resulting source model can beeffectively adapted to the target domain without having access to sourcedata, thanks to feature alignment and self-supervision. We evaluate theproposed approach in a variety of settings, considering several domainadaptation  benchmarks  and  we  show  that  our  method  is  able  to  au-tomatically  discover  relevant  semantic  information  even  in  presence  offew target samples and yields state-of-the-art results on multiple domainadaptation benchmarks. We make our source code public.*

## 2. Proposed Methods

Our ACIDS method operates in two phases: training on the source domains and adaptation to the target domain.
In the first phase, we employ an information-theoretic loss based on mutual information maximization to discover clusters in the source domains. In order to favor the emergence of clusters based on semantic information rather than domain style differences, we propose the use of a novel mutual information loss for domain alignment which is paired with BN-based feature alignment alignment.


<p align="center">
    <img src="./imgs/method.png"> <br />
    <em> 
    Figure 2. Illustration of the proposed ACIDS method for Unsupervised Clustering under Domain Shift setting. (Left) training on the source domains, (Right) adaptation to the target domain
    </em>
</p>
