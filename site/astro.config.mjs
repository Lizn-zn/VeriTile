// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://lizn-zn.github.io',
  base: '/VeriTile',
  trailingSlash: 'always',
  integrations: [
    starlight({
      title: {
        en: 'VeriTile',
        'zh-CN': 'VeriTile',
      },
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        'zh-cn': { label: '简体中文', lang: 'zh-CN' },
      },
      customCss: ['./src/styles/theme.css'],
      components: {
        Hero: './src/components/Hero.astro',
      },
      expressiveCode: {
        themes: ['github-dark-dimmed', 'github-light'],
        styleOverrides: {
          borderRadius: '2px',
          frames: {
            shadowColor: 'transparent',
          },
        },
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Lizn-zn/VeriTile',
        },
      ],
      sidebar: [
        {
          label: 'Start here',
          translations: { 'zh-CN': '从这里开始' },
          items: [
            { label: 'Overview', translations: { 'zh-CN': '概览' }, slug: 'overview' },
            { label: 'Project status', translations: { 'zh-CN': '项目状态' }, slug: 'status' },
            { label: 'Roadmap', translations: { 'zh-CN': '路线图' }, slug: 'roadmap' },
          ],
        },
        {
          label: 'Bench cookbook',
          translations: { 'zh-CN': 'Bench 翻译手册' },
          items: [{ autogenerate: { directory: 'cookbook' } }],
        },
        {
          label: 'Architecture & semantics',
          translations: { 'zh-CN': '架构与语义' },
          items: [{ autogenerate: { directory: 'architecture' } }],
        },
        {
          label: 'Proofs & surfaces',
          translations: { 'zh-CN': '证明与表面' },
          items: [{ autogenerate: { directory: 'proofs' } }],
        },
        {
          label: 'Reference',
          translations: { 'zh-CN': '参考' },
          items: [{ autogenerate: { directory: 'reference' } }],
        },
      ],
    }),
  ],
});
