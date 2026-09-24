// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://lizn-zn.github.io',
  base: '/VeriTile',
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  integrations: [
    starlight({
      title: 'VeriTile',
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
      },
      customCss: ['./src/styles/theme.css'],
      components: {
        Hero: './src/components/Hero.astro',
        Header: './src/components/Header.astro',
        Footer: './src/components/Footer.astro',
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
          items: [
            { label: 'Overview', slug: 'overview' },
            { label: 'Project status', slug: 'status' },
            { label: 'Proof coverage', slug: 'proofs/coverage' },
            { label: 'Roadmap', slug: 'roadmap' },
          ],
        },
        {
          label: 'Bench cookbook',
          items: [{ autogenerate: { directory: 'cookbook' } }],
        },
        {
          label: 'Architecture & semantics',
          items: [{ autogenerate: { directory: 'architecture' } }],
        },
        {
          label: 'Proofs & surfaces',
          items: [{ autogenerate: { directory: 'proofs' } }],
        },
        {
          label: 'Reference',
          items: [{ autogenerate: { directory: 'reference' } }],
        },
      ],
    }),
  ],
});
