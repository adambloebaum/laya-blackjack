const {test, expect} = require('@playwright/test');

test('sealed final-test audits show state counts without a training counter', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'targeted-test',status:'running',stage:'sealed_test_evaluation',elapsed_seconds:600,deadline_unix:Date.now()/1000+3600,progress:{'audit/candidate':{stage:'sdk_audit',completed:128,total:8192}}}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-progress')).toContainText('128 / 8,192 final-test states');
  await expect(page.locator('#overnight-note')).not.toContainText('Unable');
});

test('paired research progress distinguishes blocks from rounds', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'paired-test',status:'running',stage:'policy_evaluation',elapsed_seconds:600,deadline_unix:Date.now()/1000+3600,progress:{'candidate / continuous':{stage:'evaluation',mode:'continuous',completed_units:400,total_units:1000,rounds:40000}}}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-progress')).toContainText('400');
  await expect(page.locator('#overnight-progress')).toContainText('blocks');
  await expect(page.locator('#overnight-note')).toContainText('independent blocks');
  await expect(page.locator('#train-button')).toBeDisabled();
});

test('overnight progress reports both GPUs and prevents overlapping jobs', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'overnight-test',status:'running',stage:'training',elapsed_seconds:3600,deadline_unix:Date.now()/1000+3600,progress:{dataset:{stage:'generation',states:{train:100000,test:5000}},'gpu-0':{stage:'training',updates:50,planned_updates:1000,loss:0.42},'gpu-1':{stage:'training',updates:45,planned_updates:1000,loss:0.44}}}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-status')).toHaveText('TRAINING · RUNNING');
  await expect(page.locator('#overnight-progress')).toContainText('gpu-0');
  await expect(page.locator('#overnight-progress')).toContainText('gpu-1');
  await expect(page.locator('#train-button')).toBeDisabled();
  await expect(page.locator('#benchmark-button')).toBeDisabled();
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

test('table steps, reconfigures, exports and fits a phone', async ({page}) => {
  const errors=[];page.on('pageerror', e=>errors.push(e.message));
  await page.goto('/');
  await expect(page.locator('#turn-label')).toContainText('to act');
  await expect(page.locator('.seat')).toHaveCount(3);
  await page.locator('#step').click();
  await expect(page.locator('#events')).toContainText('stand');
  await page.locator('#configure').click();
  await page.locator('[name="players"]').selectOption('7');
  await page.locator('[name="samples"]').selectOption('64');
  await page.getByRole('button',{name:'Start new session'}).click();
  await expect(page.locator('.seat')).toHaveCount(7);
  const download=page.waitForEvent('download');await page.locator('#export').click();await download;
  await page.screenshot({path:'artifacts/screenshots/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await expect(page.locator('.seat')).toHaveCount(7);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:'artifacts/screenshots/mobile.png',fullPage:true});
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#train-button')).toBeVisible();
  await expect(page.locator('#benchmark-button')).toBeVisible();
  expect(errors).toEqual([]);
});

test('autoplay can pause without a runaway request loop', async ({page})=>{
  await page.goto('/');
  await expect(page.locator('#step')).toBeEnabled();
  await page.locator('#autoplay').click();
  await expect(page.locator('#autoplay')).toContainText('Pause');
  await page.locator('#autoplay').click();
  await expect(page.locator('#autoplay')).toContainText('Play');
  await expect(page.locator('#step')).toBeEnabled();
});
