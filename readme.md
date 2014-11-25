CudaLearn
=========

Advanced machine learning algorithms in C# 
-


Machine Learning is a branch of artificial intelligence focuses on the recognition of patterns and regularities in data. In many cases, these patterns are learned from labeled "training" data (supervised learning), but when no labeled data are available other algorithms can be used to discover previously unknown patterns (unsupervised learning).

Most of the high performance machine learning algorithms are written in C++ or Python. Other like Apache Mahout uses Hadoop to distribute the cost of training the models across entire clusters.

CudaLearn follows a hybrid approach. Where high performance is needed we follow the C++/Python route of implementing vertical scalability through GPU accelerated computation and aim to distribute this computation across entire clusters. This approach also exploit algorithms asymmetries in computational complexity, CPU implementations will be available for the less computationally demanded stage.

The first release will contain: 

- Deep Learning Algorithms 
  - Convolutional Neural Networks (CNN)
- Support for GPU acceleration with CUDA Kernels.
 
In the future:

- Deep Learning Algorithms 
  - Restricted Boltzmann Machines (RBM)
- Latent Dirichlet Analysis.


Licensing
-

CudaLearn is dual licensed. 

- AGPL with an explicit exception for OSS Projects 
  - You can release your project under any other OSI approved license as long as you dont change CudaLearn own licensing). However, beware that users of your project would still need to comply to the licensing terms.
  - Free support is like UDP, *Best Effort*.
- Commercial license with support guarantees.

### Commercial Use.

Commercial editions can be used in closed source environments and are available under a subscription or perpetual pricing model. While the subscription is valid, new major releases are included.
 
Developers are encouraged to buy a license early on to support the development (we know you will want to build concepts first). Once you are ready to deploy your application you will need to buy a license.
 
Licenses are available per developer, per site or per company. 

### Free (as in free beer) licenses. 

Personal, demonstrations, research, open source projects can use it for free. Essentially any non-for-profit activity.
Please contact us including a few details about your project (most importantly where is the code published) so we can let others know about it. 
 
For research activities we encourage you to cite CudaLearn in your papers.
 
Bizspark and startups are elegible for free project wide license if their revenue per year is less than 500000 USD

### Features for hire

You can ask for a quote on any specific feature your business requires to assign a dedicated team to tackle it.

Examples like:

- Recommendation engines (orders & customers pattern analysis)
- Computer Vision problems like realtime camera feed classification and recognition.
- Support for Hadoop or Azure HDInsights
- Time series analysis and prediction (ex. faults detection on sensor arrays).

And many more are a sure pick for our team to tackle.


For questions & licensing contact: <a href="mailto:cudalearn@corvalius.com?subject=[CudaLearn] Licensing">cudalearn@corvalius.com</a>



<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
CUDA is a Trademark of NVidia Corp.
