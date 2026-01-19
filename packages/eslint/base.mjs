import js from '@eslint/js'
import prettier from 'eslint-config-prettier'

/** @type {import("eslint").Linter.Config[]} */
export default [
  js.configs.recommended,
  prettier,
  {
    ignores: ['node_modules/**', 'dist/**', '.next/**', '.turbo/**'],
  },
]
