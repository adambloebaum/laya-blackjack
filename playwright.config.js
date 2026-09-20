const {defineConfig} = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests/browser',
  use: {baseURL: 'http://127.0.0.1:8000', viewport:{width:1440,height:1100}},
  webServer: {command: '.venv/bin/blackjack serve', url:'http://127.0.0.1:8000/api/health', reuseExistingServer: !process.env.CI},
});
