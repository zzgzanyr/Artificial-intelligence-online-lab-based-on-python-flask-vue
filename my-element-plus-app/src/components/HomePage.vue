<template>
  <el-container>
    <!-- 顶部导航栏 -->
    <el-header class="header">
    <el-row type="flex" align="middle" justify="space-between">
      <el-col :span="4" class="logo-container" @click="goHome">
        <img src="/Photo/logo4.png" class="logo" alt="Logo"/>
      </el-col>
      <el-col :span="200" class="header-nav">
        <el-button type="text" class="nav-button" @click="goHome" style="left: 0px;">主页</el-button>
        <el-button type="text" class="nav-button" @click="goToProjects" style="left: 100px;">项目</el-button>
        <el-button type="text" class="nav-button" @click="goToAboutPage" style="left: 200px;">产品</el-button>
        <el-button type="text" class="nav-button" @click="goToAboutPage" style="left: 300px;">博客</el-button>
        <el-button type="text" class="nav-button" style="left: 1200px;">登录</el-button>
        <el-button type="text" class="nav-button" style="left: 1300px;">注册</el-button>

      </el-col>
    </el-row>
  </el-header>

    <!-- 轮播图部分 -->
    <el-carousel :interval="5000" arrow="always" height="100vh" indicator-position="outside">
      <el-carousel-item v-for="(slide, index) in slides" :key="index">
        <section class="hero-section" :style="{ backgroundImage: `url(${slide.image})` }">
          <div class="hero-content">
            <h1 class="hero-title">{{ slide.title }}</h1>
            <p class="hero-subtitle">{{ slide.subtitle }}</p>
            <el-button type="primary" class="hero-button" @click="scrollToProjects">了解学习社区</el-button>
          </div>
        </section>
      </el-carousel-item>
    </el-carousel>

    <!-- 第二部分：项目展示区域 -->
    <el-main ref="projectsSection">
      <el-row :gutter="20">
        <el-col :span="12" v-for="(project, index) in projects" :key="index">
          <el-card shadow="always" class="project-card">
            <img :src="project.image" class="image" />
            <div class="card-content">
              <span>{{ project.title }}</span>
              <el-button type="primary" @click="goToProjectPage(index)">启动项目</el-button>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </el-main>

  <!-- 底部信息 -->
<el-footer class="footer">
  <el-row class="footer-content" >
     <el-col :span="4" class="qr-code-container" style="left: 200px;" >
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
        <li><a  class="list__item--blue"><span>商业合作</span></a></li>
        <li><a  class="list__item--blue"><span>Github</span></a></li>
        <li><a  class="list__item--blue"><span>CSDN博客  </span></a></li>
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
export default {
  name: 'HomePage',
  data() {
    return {
      slides: [
        {
          title: '开源开放，共享共进',
          subtitle: '一个在线深度强化学习社区，致力于分享、交流深度强化学习的前沿技术，推动深度强化学习技术与应用落地的学术组织。',
          image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back2.jpg')
        },
        {
          title: 'AI 深度学习项目',
          subtitle: '探索最新的深度学习技术与应用，推动技术革新。',
          image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back3.jpg')
        },
        {
          title: '加入我们的社区',
          subtitle: '与全球顶尖的AI研究人员和工程师一起成长。',
          image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back4.png')
        }
      ],
      projects: [
        { title: '卷积神经网络', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\image1.png') },
        { title: '单个水果识别', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back8.jpg') },
        { title: '多个水果识别', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back7.jpg') },
        { title: '视频处理', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\image4.png') }
      ]
    };
  },
  methods: {

    goToProjects() {
      this.$router.push('/projects');
    },
     goToProjectPage(index) {
      if (index === 1) {
        this.$router.push('/fruits');
      }
      if (index === 0){
        this.$router.push('/configurator');
      }
      if (index === 3){
        this.$router.push('/configurator');
      }
      if (index === 2){
        this.$router.push('/morefruit');
      }
    },
    goHome() {
      this.$router.push('/');
    },
    goToAboutPage() {
      window.location.href = 'https://weread.qq.com/';
    },

    scrollToProjects() {
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
.logo-title {
  font-size: 1.5em;
  font-weight: bold;
}
.header-nav {
  display: flex;
  justify-content: flex-end;
  position: relative;
}
.nav-button {
  color: white;
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
}

.hero-section {
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  height: 100vh; /* 调整图片高度 */
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
  text-align: center;
  position: relative;
  padding: 20px;
}
.hero-section::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.2); /* 半透明黑色渐变 */
}
.hero-content {
  position: relative;
  z-index: 1;
  padding: 200px;
  border-radius: 100px;
}
.hero-title {
  font-size: 3em;
  color: #ffcc00; /* 设置 h1 文字颜色 */
  margin-bottom: 0.5em;
}
.hero-subtitle {
  font-size: 1.5em;
  color: #ffffff; /* 设置 p 文字颜色 */
  margin-bottom: 1em;
}
.hero-button {
  font-size: 1.2em;
  padding: 10px 20px;
}

.project-card {
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 20px;
}
.image {
  width: 100%;
  height: 200px;
  object-fit: cover;
}
.card-content {
  padding: 14px;
  text-align: center;
}
.card-content span {
  display: block;
  margin-bottom: 10px;
  font-size: 18px;
  font-weight: bold;
}
.card-content .el-button {
  width: 100%;
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