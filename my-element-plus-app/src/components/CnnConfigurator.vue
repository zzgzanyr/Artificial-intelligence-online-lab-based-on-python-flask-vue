<template>
  <el-container class="model-config">
    <!-- 顶部导航栏 -->
   <el-header class="header">
    <el-row type="flex" align="middle" justify="space-between">
      <el-col :span="4" class="logo-container" @click="goHome">
        <img src="/Photo/logo4.png" class="logo" alt="Logo"/>
      </el-col>
      <el-col :span="200" class="header-nav">
        <el-button type="text" class="nav-button" @click="goToProjects" style="position: absolute;left: 150px;">项目</el-button>
        <el-button type="text" class="nav-button" @click="goHome" style="left: 300px;">主页</el-button>
        <el-button type="text" class="nav-button" @click="goToAboutPage" style="position: absolute;left: 280px;">产品</el-button>
        <el-button type="text" class="nav-button" @click="goToAboutPage" style="position: absolute;left: 400px;">博客</el-button>
        <el-button type="text" class="nav-button" style="position: absolute;left: 1250px;">登录</el-button>
        <el-button type="text" class="nav-button" style="position: absolute;left: 1300px;">注册</el-button>
      </el-col>
    </el-row>
  </el-header>

    <div class="cnn-introduction">
    <h2>卷积神经网络介绍</h2>
    <p>卷积神经网络（Convolutional Neural Network，CNN）是一种深度学习模型，被广泛用于图像识别和计算机视觉任务。它模仿人类视觉系统的工作原理，通过多层卷积和池化层提取图像特征，
    并通过全连接层进行分类或回归。</p>
    <img src="/Photo/cnn-structure.jpg" alt="CNN Structure" class="cnn-structure">
    <p> 如图所示，输入是一张手写字母“A”的图像。输入层的尺寸为32x32，表示输入图像的分辨率为32x32像素。第一层卷积层（C1）产生了6个28x28的特征图。
    卷积层通过卷积核在输入图像上滑动，提取局部特征。第一层池化层（S2）对每个特征图进行下采样，
    通常是最大池化或平均池化。这里将特征图尺寸缩小到14x14。池化操作有助于减少数据的维度和计算量，同时保留重要的特征。第二层卷积层（C3）产生了16个10x10的特征图，
    再次通过卷积操作提取更高层次的特征。第二层池化层（S4）进一步对特征图进行下采样，将尺寸缩小到5x5。第五层（C5）是一个全连接层，包含120个神经元。
    这一层将池化层输出的特征图展平成一个一维向量，并通过全连接层进一步处理。第六层（F6）也是一个全连接层，包含84个神经元。这个层次进一步对数据进行处理和分类。
    输出层包含10个神经元，对应于分类任务中的10个类别（例如，0到9的手写数字识别）。输出层的结果即为模型的预测结果。</p>
  </div>
    <el-main>
      <!-- 其他按钮和功能 -->
      <el-form-item>
        <el-button type="primary" @click="triggerModelConfigModal">训练模型</el-button>
        <el-button type="primary" @click="triggerFileUpload">导入模型</el-button>
        <input type="file" ref="fileInput" @change="loadModel" style="display: none" />
      </el-form-item>

      <el-dialog v-model="isConfigModalVisible" title="配置模型" width="30%">
        <el-form :model="modelConfig" label-width="180px" v-loading="loading" element-loading-text="Loading...">
          <el-form-item label="卷积层数">
            <el-input v-model.number="modelConfig.conv_layers" type="number"></el-input>
          </el-form-item>
          <el-form-item label="Filters">
            <el-input v-model.number="modelConfig.filters" type="number"></el-input>
          </el-form-item>
          <el-form-item label="Kernel Size">
            <el-input v-model.number="modelConfig.kernel_size" type="number"></el-input>
          </el-form-item>
          <el-form-item label="Pool Size">
            <el-input v-model.number="modelConfig.pool_size" type="number"></el-input>
          </el-form-item>
          <el-form-item label="Dense Units">
            <el-input v-model.number="modelConfig.dense_units" type="number"></el-input>
          </el-form-item>
          <el-form-item label="Epochs">
            <el-input v-model.number="modelConfig.epochs" type="number"></el-input>
          </el-form-item>
        </el-form>
        <template v-slot:footer>
          <el-button @click="isConfigModalVisible = false">取消</el-button>
          <el-button type="primary" @click="submitModelConfig">确认</el-button>
        </template>
      </el-dialog>

      <!-- 显示训练进展的对话框 -->

        <canvas ref="chart"></canvas>


      <!-- Training Result and Save Model -->
      <el-alert
        v-if="trainResult"
        title="Training Result"
        type="success"
        :description="'模型准确率: ' + (trainResult.accuracy * 100).toFixed(2) + '%'"
        show-icon>
      </el-alert>
      <el-form v-if="trainResult" label-width="180px">
        <el-form-item label="Save Model Path">
          <el-input v-model="savePath" placeholder="Enter path to save model"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="saveModel">Save Model</el-button>
        </el-form-item>
      </el-form>

      <!-- Model Testing -->
      <el-form v-if="modelLoaded || trainResult" label-width="180px">
        <el-form-item>
          <el-button type="primary" @click="triggerImageUpload">测试模型</el-button>
          <input type="file" ref="imageInput" @change="testModel" style="display: none" />
        </el-form-item>
        <el-form-item v-if="selectedImage" class="image-container">
          <img :src="selectedImage" alt="Selected Image" class="selected-image" />
        </el-form-item>
      </el-form>
      <el-alert
        v-if="validationResult"
        title="Validation Result"
        type="info"
        :description="'Prediction: ' + validationResult.prediction"
        show-icon>
      </el-alert>
    </el-main>

    <el-footer class="footer">
      <el-row class="footer-content">
        <el-col :span="4" class="qr-code-container" style="left: 200px;">
          <img src="/Photo/3code.png" alt="QR Code" class="qr-code">
        </el-col>
        <el-col :span="2" class="footer-links-column" style="position: absolute; left: 680px;">
          <ul class="footer-links">
            <li><a class="list__item--blue"><span>社区规则</span></a></li>
            <li><a class="list__item--blue"><span>捐赠</span></a></li>
            <li><a class="list__item--blue"><span>反馈</span></a></li>
            <li><a class="list__item--blue"><span>公众号</span></a></li>
          </ul>
        </el-col>
        <el-col :span="2" class="footer-links-column" style="position: absolute; left: 780px;">
          <ul class="footer-links">
            <li><a class="list__item--blue"><span>关于我们</span></a></li>
            <li><a class="list__item--blue"><span>商业合作</span></a></li>
            <li><a class="list__item--blue"><span>Github</span></a></li>
            <li><a class="list__item--blue"><span>CSDN博客</span></a></li>
          </ul>
        </el-col>
      </el-row>
      <div class="footer-divider"></div>
      <p class="footer-note">
        本网站页面内容仅供学生分享和交流使用，如有侵权，请立即联系我们，我们将在24小时内进行处理和解决
      </p>
      <p class="footer-copy">
        Copyright © 2019-2023 梦溪AI实验室 All Rights Reserved
      </p>
    </el-footer>
  </el-container>
</template>


<script>
import axios from 'axios';
import { ref, nextTick } from 'vue';
import Chart from 'chart.js/auto';

export default {
  name: 'CnnConfigurator',
  setup() {
    const modelConfig = ref({
      conv_layers: 2,
      filters: 32,
      kernel_size: 3,
      pool_size: 2,
      dense_units: 128,
      epochs: 5
    });

    const loading = ref(false);
    const isConfigModalVisible = ref(false);
    const isResultModalVisible = ref(false);
    const trainResult = ref(null);
    const validationResult = ref(null);
    const savePath = ref('');
    const selectedImage = ref('');
    const modelLoaded = ref(false);
    const fileInput = ref(null);
    const imageInput = ref(null);
    const chartInstance = ref(null);

    const triggerModelConfigModal = () => {
      isConfigModalVisible.value = true;
    };

    const submitModelConfig = async () => {
      loading.value = true;
      try {
        const response = await axios.post('http://localhost:5000/train', modelConfig.value);
        trainResult.value = response.data;
        modelLoaded.value = false;

        // 显示结果弹窗并绘制图表
        isResultModalVisible.value = true;
        await nextTick(); // 确保 DOM 更新后再绘制图表
        renderChart(response.data.history.accuracy);
      } catch (error) {
        console.error('Error:', error);
        alert('Failed to train the model. See console for details.');
      }
      loading.value = false;
      isConfigModalVisible.value = false;
    };

    const renderChart = (data) => {
      if (chartInstance.value) {
        chartInstance.value.destroy();
      }
      const ctx = document.querySelector("canvas").getContext("2d");
      chartInstance.value = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map((_, index) => `Epoch ${index + 1}`),
          datasets: [{
            label: '准确率',
            data: data,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 2,
            fill: false
          }]
        },
        options: {
          responsive: true,
          scales: {
            x: {
              title: {
                display: true,
                text: 'Epoch'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Accuracy'
              },
              beginAtZero: true
            }
          }
        }
      });
    };

    const triggerFileUpload = async () => {
      await nextTick();
      fileInput.value.click();
    };

    const loadModel = async (event) => {
      const file = event.target.files[0];
      if (!file) {
        alert('Please select a model file.');
        return;
      }
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post('http://localhost:5000/load', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        if (response.data.error) {
          alert(`Error: ${response.data.error}`);
        } else {
          trainResult.value = response.data;
          modelLoaded.value = true;
          alert('Model loaded successfully.');
        }
      } catch (error) {
        console.error('Error:', error);
        alert('Failed to load the model. See console for details.');
      }
    };

    const saveModel = async () => {
      try {
        await axios.post('http://localhost:5000/save', { path: savePath.value });
        alert('Model saved successfully.');
      } catch (error) {
        console.error('Error:', error);
        alert('Failed to save the model. See console for details.');
      }
    };

    const triggerImageUpload = async () => {
      await nextTick();
      imageInput.value.click();
    };

    const testModel = async (event) => {
      const file = event.target.files[0];
      if (!file) {
        alert('Please select an image.');
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        selectedImage.value = e.target.result;
      };
      reader.readAsDataURL(file);

      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post('http://localhost:5000/validate', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        validationResult.value = response.data;
      } catch (error) {
        console.error('Error:', error);
        alert('Failed to validate the model. See console for details.');
      }
    };

    //const goHome = () => {
     // router.push('/');
    //};

    //const goToProjects = () => {
     //  router.push(`/configurator`);
    //};
    //const goToAboutPage = ()=> {
     // window.location.href = 'https://weread.qq.com/';
    //};

    return {
      loading,
      modelConfig,
      trainResult,
      validationResult,
      savePath,
      selectedImage,
      modelLoaded,
      submitModelConfig,
      loadModel,
      saveModel,
      triggerFileUpload,
      triggerImageUpload,
      testModel,
      fileInput,
      imageInput,
      //goHome,
      //goToProjects,
      isConfigModalVisible,
      triggerModelConfigModal,
      //goToAboutPage,
      isResultModalVisible
    };
  },
  methods: {
    goToProjects() {
      this.$router.push('/projects');
    },
    goToProjectPage() {
      this.$router.push(`/configurator`);
    },
    goHome() {
      this.$router.push('/');
    },
    goToAboutPage() {
      window.location.href = 'https://weread.qq.com/';
    },

    scrollToProjects() {
      const projectsSection = this.$refs.projectsSection;
      if (projectsSection) {
        projectsSection.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }
};
</script>



<style scoped>
.header {
  background-color: #333333;
  color: white;
  padding: 10px 20px;
  display: flex;
  align-items: center;
}
.logo-container {
  display: flex;
  align-items: center;
}
.logo {
  height: 50px;
  margin-right: 10px;
}
.header-nav {
  display: flex;
  justify-content: space-around;
  flex: 1;
}
.nav-button {
  color: white;
}

.model-config {
  max-width: 6000px;
  margin: 0 auto;
  padding: 0px;
  text-align: center;
}
.el-form-item {
  margin-bottom: 20px;
}
.selected-image {
  max-width: 100%;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 10px;
  background: #fff;
}
.image-container {
  text-align: center;
}

.footer {
  height: 400px; /* 设置固定高度，可以根据需要调整 */
  background-color: #333; /* 背景颜色 */
 color: white; /* 文本颜色 */
  text-align: center; /*文本居中对齐 */
  display: flex;
  flex-direction: column;
  justify-content: center; /* 垂直居中对齐 */
  align-items: center; /* 水平居中对齐 */
}
.footer-content {
  width: 100%; /* 让行元素占满宽度 */
  margin-bottom: 20px; /* 每个部分之间留出一些空间 */
}
.footer h3 {
  margin-top: 0;
}
.qr-code-container {
  display: flex;
  /*justify-content: center;*/
  /*align-items: center;*/
  margin-bottom: 0px;
}
.qr-code {
  max-width: 100px;
}
.footer-links {
  list-style: none;
  padding: 0;
}
.footer-links li {
  margin: 5px 0;
}
/*
.footer-links a {
  color: white;
  text-decoration: none;
}*/
/*
.footer-links a:hover {
  text-decoration: underline;
}*/
.footer-links-column {
  padding-left: 0px; /* 缩小列之间的距离 */
  padding-right: 0px; /* 缩小列之间的距离 */
}
.footer-divider {
  width: 50%; /* 设置横线宽度为页面宽度的一半 */
  border-top: 1px solid white; /* 设置横线样式 */
  margin: 30px auto; /* 居中对齐，并留出上下空白 */
}
.footer-note, .footer-copy {
  margin-top: 20px;
}

</style>
<style lang="scss" scoped>
body {
  display: table;
  width: 100%;
  height: 100vh;
  margin: 0;
  background: #222;
  font-family: 'Roboto Condensed', sans-serif;
  font-size: 26px;
  font-weight: 600;
  letter-spacing: 5px;
  text-transform: uppercase;
}

.container {
  display: table-cell;
  vertical-align: middle;
  text-align: center;
  width: 600px;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
}

li {
  display: inline-block;
  padding: 0 20px;
}

span {
  position: relative;
  display: block;
  cursor: pointer;
}

.list__item--yellow {
  color: #FFC56C;
}
.list__item--blue {
  color: #6EC5E9;
  /* 6EC5E9*/
}

.list__item--red {
  color: #FF5959;
}

span {
  &:before, &:after {
    content: '';
    position: absolute;
    width: 0%;
    height: 4px;
    bottom: -2px;
    margin-top: -0.5px;
    background: #fff;
  }

  &:before {
    left: -2.5px;
  }

  &:after {
    right: 2.5px;
    background: #fff;
    transition: width 0.8s cubic-bezier(0.22, 0.61, 0.36, 1);
  }

  &:hover {
    &:before {
      background: #fff;
      width: 100%;
      transition: width 0.5s cubic-bezier(0.22, 0.61, 0.36, 1);
    }

    &:after {
      background: transparent;
      width: 100%;
      transition: 0s;
    }
  }
}
</style>

<style scoped>
.loading-bar {
  width: 100%;
  height: 4px;
  background-color: #90EE90;
  animation: loading 2s linear infinite;
}
@keyframes loading {
  0% { width: 0%; }
  100% { width: 100%; }
}
</style>