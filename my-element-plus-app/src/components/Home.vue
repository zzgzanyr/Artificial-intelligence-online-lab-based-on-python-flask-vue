<template>
  <el-container>
    <el-header>
      <h1>Welcome to the CNN Configurator</h1>
    </el-header>
    <el-main>
      <el-button type="primary" @click="goToConfigurator">Go to Configurator</el-button>
      <el-button type="info" @click="goToAbout">About</el-button>
    </el-main>
  </el-container>
</template>

<script>
export default {
  name: 'HomePage', // 修改组件名称
  methods: {
    goToConfigurator() {
      this.$router.push('/configurator');
    },
    goToAbout() {
      this.$router.push('/about');
    }
  }
};
</script>

<style>
/* 样式部分可根据需要进行调整 */
</style>

