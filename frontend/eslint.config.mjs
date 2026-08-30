// ESLint flat config -- adaptee du standard barkahub (packages/config), version standalone Vite.
// Le linter est le reviewer : type-aware strict + stylistic + tri des imports + hooks/a11y React.
// Ecarts assumes vs barkahub : pas de regle RTL (app FR en LTR), pas de specificites Nx/monorepo.
import js from '@eslint/js';
import stylistic from '@stylistic/eslint-plugin';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import perfectionist from 'eslint-plugin-perfectionist';
import reactHooks from 'eslint-plugin-react-hooks';
import unicorn from 'eslint-plugin-unicorn';
import unusedImports from 'eslint-plugin-unused-imports';
import globals from 'globals';
import tseslint from 'typescript-eslint';

// Naming aligne sur le bloc "web" de barkahub : composants et proprietes-composant en PascalCase.
const NAMING = [
  'error',
  { selector: 'default', format: ['camelCase'], leadingUnderscore: 'allow' },
  {
    selector: 'variable',
    format: ['camelCase', 'PascalCase', 'UPPER_CASE'],
    leadingUnderscore: 'allow',
  },
  { selector: 'function', format: ['camelCase', 'PascalCase'] },
  { selector: 'parameter', format: ['camelCase', 'PascalCase'], leadingUnderscore: 'allow' },
  { selector: 'typeLike', format: ['PascalCase'] },
  { selector: 'import', format: ['camelCase', 'PascalCase'] },
  { selector: ['objectLiteralProperty', 'typeProperty'], format: null },
];

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      globals: { ...globals.browser },
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    plugins: {
      '@stylistic': stylistic,
      perfectionist,
      unicorn,
      'unused-imports': unusedImports,
    },
    rules: {
      // Idiomes TypeScript
      '@typescript-eslint/consistent-type-definitions': ['error', 'type'],
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/prefer-nullish-coalescing': 'error',
      'no-else-return': 'error',

      // Bans : enums (utiliser 'as const' + unions)
      'no-restricted-syntax': [
        'error',
        {
          selector: 'TSEnumDeclaration',
          message: "Enums bannis. Utiliser 'as const' + types union.",
        },
      ],

      // Hygiene
      'no-console': ['error', { allow: ['warn', 'error'] }],
      'no-empty': 'error',
      curly: ['error', 'all'],

      // Nommage + fichiers en kebab-case
      '@typescript-eslint/naming-convention': NAMING,
      'unicorn/filename-case': ['error', { case: 'kebabCase' }],

      // Imports : ordre, groupes, suppression du code mort
      'perfectionist/sort-imports': ['error', { type: 'natural', newlinesBetween: 1 }],
      'unused-imports/no-unused-imports': 'error',
      '@typescript-eslint/no-unused-vars': 'off',
      'no-unused-vars': 'off',
      'unused-imports/no-unused-vars': [
        'warn',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],

      // Lisibilite
      '@stylistic/lines-around-comment': [
        'error',
        {
          beforeLineComment: true,
          beforeBlockComment: true,
          allowBlockStart: true,
          allowObjectStart: true,
          allowArrayStart: true,
          allowClassStart: true,
          allowInterfaceStart: true,
          allowTypeStart: true,
        },
      ],
      '@stylistic/no-extra-semi': 'error',

      // Limites souples (warn)
      'max-lines-per-function': ['warn', { max: 120, skipBlankLines: true, skipComments: true }],
      'max-lines': ['warn', { max: 400, skipBlankLines: true, skipComments: true }],
      complexity: ['warn', 15],
    },
  },

  // Code applicatif React : regles react-hooks + jsx-a11y.
  {
    files: ['src/**/*.{ts,tsx}'],
    plugins: { 'jsx-a11y': jsxA11y, 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs['recommended-latest'].rules,
      ...jsxA11y.flatConfigs.recommended.rules,
    },
  },

  // Fichiers .d.ts : `interface` requis pour l'augmentation de modules/globaux
  // (ex. ImportMetaEnv merge avec vite/client), ce que `type` ne peut pas faire.
  {
    files: ['**/*.d.ts'],
    rules: { '@typescript-eslint/consistent-type-definitions': 'off' },
  },

  // Fichiers de config JS : pas de lint type-aware.
  {
    files: ['**/*.{js,cjs,mjs}'],
    extends: [tseslint.configs.disableTypeChecked],
  },
);
