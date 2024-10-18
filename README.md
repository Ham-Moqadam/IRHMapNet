# Going deeper with deep learning: automatically tracing1 internal reflection horizons in ice sheets

## a deep learning framework for automatic tracing of IRHs from RES radargrams. 

### Hameed Moqadam


Mapping the internal stratigraphy of ice sheets is crucial for various glaciological applications, including the study of past ice flows, current surface mass balance distributions, melting processes, and contemporary ice dynamics. These elements are vital for enhancing future projections of sea level rise. The predominant method for investigating the internal structure of ice sheets is radio-echo sounding (RES). Traditional approaches for mapping englacial stratigraphy have largely relied on time-consuming manual or semi-automatic methods. Although effective, these approaches are not feasible for the comprehensive analysis of the extensive data available.

In recent years, there has been a growing interest in (semi-)automatic mapping of internal stratigraphy from RES radargrams, supported by advances in machine learning. This paper introduces IRHMapNet, a deep learning framework that leverages a U-Net-based architecture for the automatic tracing of internal reflection horizons (IRHs) from radargrams. The framework is built upon airborne RES data, incorporating both pre-processing and post-processing methods.

To train the U-Net architecture, we utilize a combination of fully mapped hand-labelled data, results from image processing and thresholding methods, and layer slope inference. Our evaluation demonstrates the successful performance of IRHMapNet in tracing IRHs, including for deep and extended analyses. We present various metrics to assess the model's effectiveness and discuss remaining challenges and limitations associated with machine learning approaches. The results indicate that IRHMapNet effectively meets its objectives, showcasing the potential of deep learning in the automated analysis of radar data.
