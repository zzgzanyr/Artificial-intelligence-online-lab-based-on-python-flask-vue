import { createRouter, createWebHistory } from 'vue-router';
import HomePage from './components/HomePage.vue';
import CnnConfigurator from './components/CnnConfigurator.vue';
import AboutPage from './components/AboutPage.vue';
import ProjectsPage from './components/ProjectsPage.vue';
import Fruits from './components/Fruits.vue'
import Morefruit from './components/Morefruit.vue'
const routes = [
  { path: '/', component: HomePage },
  { path: '/configurator', component: CnnConfigurator },
  { path: '/about', component: AboutPage },
  {path: '/projects',
      component: ProjectsPage
    },
    {path: '/fruits', component: Fruits},
     {path: '/morefruit', component: Morefruit}
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
