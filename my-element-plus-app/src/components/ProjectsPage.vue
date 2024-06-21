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
          <el-button type="text" class="nav-button" style="left: 1250px;">登录</el-button>
          <el-button type="text" class="nav-button" style="left: 1300px;">注册</el-button>
        </el-col>
      </el-row>
    </el-header>

    <!-- 项目展示区域 -->
    <el-main>
      <el-row :gutter="20">
        <el-col :span="12" v-for="(project, index) in projects" :key="index">
          <el-card shadow="always" class="project-card">
            <img :src="project.image" class="image" />
            <div class="card-content">
              <span>{{ project.title }}</span>
              <el-button type="primary" @click="goToProject(index)">启动项目</el-button>
            </div>
          </el-card>
        </el-col>
      </el-row>
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
        本网站页面内容仅供学生分享交流使用，如有侵权，请立即联系我们，我们将在24小时内进行处理和解决
      </p>
      <p class="footer-copy">
        Copyright © 2019-2023 梦溪AI实验室 All Rights Reserved
      </p>
    </el-footer>
  </el-container>
</template>

<script>
export default {
  name: 'ProjectsPage',
  data() {
    return {
      projects: [
        { title: '卷积神经网络', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\image1.png') },
        { title: '单个水果识别', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back7.jpg') },
        { title: '多个水果识别', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\back7.jpg') },
        { title: '视频处理', image: require('C:\\Users\\17927\\Desktop\\pythonProject2\\my-element-plus-app\\Photo\\image4.png') }
      ]
    };
  },
  methods: {
    goHome() {
      this.$router.push('/');
    },
    goToProjects() {
      this.$router.push('/projects');
    },
    goToAboutPage() {
      window.location.href = 'https://weread.qq.com/';
    },
    goToProject(index) {
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