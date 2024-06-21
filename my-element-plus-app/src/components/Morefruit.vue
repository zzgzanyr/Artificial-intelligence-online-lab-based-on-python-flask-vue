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
      <h2>基于支持向量机（SVM）的机器学习方法识别水果种类</h2>
      <p>
        支持向量机（Support Vector Machine, SVM）是一类按监督学习（supervised learning）方式对数据进行二元分类的广义线性分类器（generalized linear classifier），其决策边界是对学习样本求解的最大边距超平面（maximum-margin hyperplane)。
        SVM使用铰链损失函数（hinge loss）计算经验风险（empirical risk）并在求解系统中加入了正则化项以优化结构风险（structural risk），是一个具有稀疏性和稳健性的分类器。SVM可以通过核方法（kernel method）进行非线性分类，是常见的核学习（kernel learning）方法之一。
      </p>
      <img src="/Photo/back6.jpg" alt="CNN Structure" class="cnn-structure">
      <p>
        如图所示，首先，收集包含不同水果的彩色图像。示例图像包括茄子、橙子和香蕉。将彩色图像转换为灰度图像，以减少计算复杂度和数据量。这一步将图像从RGB空间转换为灰度空间，只保留图像的亮度信息。然后使用边缘检测算法（例如Canny边缘检测）提取图像中的边缘信息。边缘检测有助于识别和定位水果的轮廓，从而更容易进行后续处理。
      </p>
      <p>
        通过形态学操作（如腐蚀和膨胀）对边缘检测结果进行处理。这些操作可以帮助去除噪声，填补边缘间的空隙，得到更清晰的目标轮廓。从处理后的图像中提取关键特征，例如形状、纹理和颜色等。然后使用支持向量机（SVM）进行分类。SVM是一种监督学习模型，通过寻找最优超平面来最大化类别间的间隔，从而实现分类任务。
      </p>
    </div>

    <el-main>
      <!-- 其他按钮和功能 -->
      <el-form-item>
        <el-button type="primary" @click="triggerModelConfigModal">训练模型</el-button>
        <el-button type="primary" @click="triggerFileUpload('model')">上传模型文件</el-button>
        <el-button type="primary" @click="triggerFileUpload('scaler')">上传标量文件</el-button>
        <el-button type="primary" @click="triggerFileUpload('labelEncoder')">上传标签编码器文件</el-button>
        <input type="file" ref="fileInputModel" @change="handleFileLoad($event, 'http://localhost:5000/fruit/upload_model')" style="display: none" />
        <input type="file" ref="fileInputScaler" @change="handleFileLoad($event, 'http://localhost:5000/fruit/upload_scaler')" style="display: none" />
        <input type="file" ref="fileInputLabelEncoder" @change="handleFileLoad($event, 'http://localhost:5000/fruit/upload_label_encoder')" style="display: none" />
      </el-form-item>

      <el-dialog v-model="isConfigModalVisible" title="配置模型" width="30%">
        <el-form :model="modelConfig" label-width="180px" v-loading="loading" element-loading-text="Loading...">
          <el-form-item label="数据路径">
            <el-input v-model="modelConfig.data_path" :disabled="true"></el-input>
          </el-form-item>
          <el-form-item label="选择类别">
            <el-select v-model="modelConfig.categories" multiple placeholder="请选择类别">
              <el-option
                v-for="item in categoryOptions"
                :key="item"
                :label="item"
                :value="item"
              ></el-option>
            </el-select>
          </el-form-item>
        </el-form>
        <template v-slot:footer>
          <el-button @click="isConfigModalVisible = false">取消</el-button>
          <el-button type="primary" @click="submitModelConfig">确认</el-button>
        </template>
      </el-dialog>

      <!-- Training Result and Save Model -->
      <el-alert
        v-if="trainResult"
        title="Training Result"
        type="success"
        :description="'模型准确率: ' + (trainResult.accuracy * 100).toFixed(2) + '%'"
        show-icon>
      </el-alert>
      <el-form v-if="trainResult" label-width="180px">
        <el-form-item label="保存模型路径">
          <el-input v-model="savePath" placeholder="Enter path to save model"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="saveModel">保存模型</el-button>
        </el-form-item>
      </el-form>

      <!-- Model Testing -->
      <el-form v-if="modelLoaded" label-width="180px">
        <el-form-item>
          <el-button type="primary" @click="triggerImageUpload">测试模型</el-button>
          <input type="file" ref="imageInput" @change="testModel" style="display: none" />
        </el-form-item>
        <el-form-item v-if="selectedImage" class="image-container">
          <img :src="selectedImage" alt="Selected Image" class="selected-image" />
          <img :src="annotatedImage" alt="Annotated Image" class="annotated-image" />
        </el-form-item>
      </el-form>
      <el-alert
        v-if="validationResult"
        title="Validation Result"
        type="info"
        :description="'Prediction: ' + validationResult.predictions.map(p => `${p[0]} (${(p[1] * 100).toFixed(2)}%)`).join(', ')"
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

export default {
  name: 'FruitClassifier',
  setup() {
    const modelConfig = ref({
      data_path: 'E://Fruit-Images-Dataset-master//Training',
      categories: []
    });

    const categoryOptions = ref([
      'Apple Braeburn',
      'Banana',
      'Watermelon',
      'Cantaloupe2',
      'Eggplant',
      'Lychee'
    ]);
    const fileInputModel = ref(null);
    const fileInputScaler = ref(null);
    const fileInputLabelEncoder = ref(null);
    const loading = ref(false);
    const isConfigModalVisible = ref(false);
    const isResultModalVisible = ref(false);
    const trainResult = ref(null);
    const validationResult = ref(null);
    const savePath = ref('');
    const selectedImage = ref('');
    const annotatedImage = ref('');
    const modelLoaded = ref(false);
    const uploadedFiles = ref({
      model: false,
      scaler: false,
      labelEncoder: false
    });
    const imageInput = ref(null);

    const checkModelLoaded = () => {
      modelLoaded.value = uploadedFiles.value.model && uploadedFiles.value.scaler && uploadedFiles.value.labelEncoder;
    };

    const triggerFileUpload = async (type) => {
      await nextTick();
      console.log(`Triggering file upload for type: ${type}`);
      if (type === 'model') {
        fileInputModel.value.click();
      } else if (type === 'scaler') {
        fileInputScaler.value.click();
      } else if (type === 'labelEncoder') {
        fileInputLabelEncoder.value.click();
      }
    };

    const handleFileLoad = (event, uploadUrl) => {
      const file = event.target.files[0];
      if (!file) {
        alert('请选择文件');
        console.log("No file selected");
        return;
      }

      console.log(`Uploading file to ${uploadUrl}`);
      const formData = new FormData();
      formData.append('file', file);

      axios.post(uploadUrl, formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      })
      .then(response => {
        console.log("Upload successful", response.data);
        alert(response.data.message);

        // 设置上传标志
        if (uploadUrl.includes('upload_model')) {
          uploadedFiles.value.model = true;
        } else if (uploadUrl.includes('upload_scaler')) {
          uploadedFiles.value.scaler = true;
        } else if (uploadUrl.includes('upload_label_encoder')) {
          uploadedFiles.value.labelEncoder = true;
        }
        checkModelLoaded();
      })
      .catch(error => {
        console.error('Error:', error);
        alert('文件上传失败，请查看控制台了解详情');
      });
    };

    const triggerModelConfigModal = () => {
      isConfigModalVisible.value = true;
    };

    const submitModelConfig = async () => {
      if (modelConfig.value.categories.length === 0) {
        alert('请选择至少一个类别');
        return;
      }
      loading.value = true;
      try {
        const response = await axios.post('http://localhost:5000/fruit/train', modelConfig.value);
        trainResult.value = response.data;
        modelLoaded.value = true; // 模型训练成功后，设置 modelLoaded 为 true
        isResultModalVisible.value = true;
        alert('模型训练成功');
      } catch (error) {
        console.error('Error:', error);
        alert('模型训练失败，请查看控制台了解详情');
      }
      loading.value = false;
      isConfigModalVisible.value = false;
    };

    const saveModel = async () => {
      if (!trainResult.value) {
        alert('请先训练模型');
        return;
      }
      if (!savePath.value) {
        alert('请输入保存模型的路径');
        return;
      }
      try {
        const response = await axios.post('http://localhost:5000/save_svm', {
          save_path: savePath.value
        }, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
        if (response.data.error) {
          alert(`保存模型失败: ${response.data.error}`);
        } else {
          alert('模型保存成功');
        }
      } catch (error) {
        console.error('Error:', error);
        alert('保存模型时发生错误，请查看控制台了解详情');
      }
    };

    const triggerImageUpload = async () => {
      await nextTick();
      imageInput.value.click();
    };

    const testModel = async (event) => {
      const file = event.target.files[0];
      if (!file) {
        alert('请选择一张图片');
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
        const response = await axios.post('http://localhost:5000/fruit/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        validationResult.value = response.data;
        annotatedImage.value = `data:image/png;base64,${response.data.image}`; // 使用正确的 Base64 编码字符串
        alert('图片预测成功');
      } catch (error) {
        console.error('Error:', error);
        alert('图片预测失败，请查看控制台了解详情');
      }
    };

    return {
      loading,
      modelConfig,
      categoryOptions,
      trainResult,
      validationResult,
      savePath,
      selectedImage,
      annotatedImage,
      modelLoaded,
      submitModelConfig,
      saveModel,
      triggerFileUpload,
      triggerImageUpload,
      testModel,
      fileInputModel,
      fileInputScaler,
      fileInputLabelEncoder,
      handleFileLoad,
      imageInput,
      isConfigModalVisible,
      triggerModelConfigModal,
      isResultModalVisible
    };
  },
  methods: {
    goToProjects() {
      this.$router.push('/projects');
    },
    goHome() {
      this.$router.push('/');
    },
    goToAboutPage() {
      window.location.href = 'https://weread.qq.com/';
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
<style>
  .cnn-introduction {
    padding: 20px;
    line-height: 1.6;  /* 增加行间距 */
  }
  .cnn-introduction h2 {
    margin-bottom: 20px;
  }
  .cnn-introduction p {
    margin-bottom: 20px;  /* 增加段落间距 */
    text-align: justify;  /* 两端对齐文本 */
  }
  .cnn-structure {
    display: block;
    margin: 20px auto;
    max-width: 100%;
    max-height: 500px;  /* 调整最大高度 */
    width: auto;
    height: auto;
  }
</style>
