const {test, expect} = require('@playwright/test');

test('matched training names both arms and keeps assembly separate from final tests', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'matched-test',experiment:'visitation-training',status:'running',stage:'training',elapsed_seconds:3600,deadline_unix:Date.now()/1000+3600,progress:{'assembly/progress.json':{stage:'replacement_labeling',completed:800,total:800,unit:'paired trajectory groups',states:32000},'training/control/progress.json':{stage:'training',updates:50,planned_updates:36000},'training/visited/progress.json':{stage:'training',updates:50,planned_updates:36000}}}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-progress')).toContainText('Control · GPU 0');
  await expect(page.locator('#overnight-progress')).toContainText('Model-visited · GPU 1');
  await expect(page.locator('#overnight-progress')).toContainText('32,000 sampled states');
  await expect(page.locator('#overnight-note')).toContainText('same update schedule');
  await expect(page.locator('#train-button')).toBeDisabled();
});

test('visitation pilot distinguishes development samples from final tests', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'pilot-test',experiment:'visitation-pilot',status:'running',stage:'qualifying_pilot_labels',elapsed_seconds:60,deadline_unix:Date.now()/1000+3600,progress:{'audit-laya':{stage:'audit',completed:3,total:100,unit:'trajectory groups',states:60}}}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-progress')).toContainText('3 / 100 trajectory groups');
  await expect(page.locator('#overnight-progress')).toContainText('60 sampled states');
  await expect(page.locator('#overnight-progress')).not.toContainText('final-test');
  await expect(page.locator('#overnight-note')).toContainText('does not train');
  await expect(page.locator('#train-button')).toBeDisabled();
});

test('completed pilot does not imply that qualification passed', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'pilot-test',experiment:'visitation-pilot',status:'complete',stage:'verifying_pilot',elapsed_seconds:60,deadline_unix:Date.now()/1000+3600,progress:{},qualified_for_training_design:false,decision:'revise_pilot'}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-note')).toContainText('qualification checks need review');
  await expect(page.locator('#overnight-note')).not.toContainText('data qualified');
});

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

test('release selection progress is distinct from final-test inference', async ({page})=>{
  await page.route('**/api/overnight', route=>route.fulfill({json:{available:true,run:'release-test',experiment:'final-release',status:'running',stage:'candidate_selection',elapsed_seconds:600,deadline_unix:Date.now()/1000+3600,progress:{'selection/incumbent/progress.json':{stage:'sdk_audit',completed:128,total:8192,split:'selection'}}}}));
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#overnight-progress')).toContainText('128 / 8,192 selection states');
  await expect(page.locator('#overnight-progress')).not.toContainText('final-test states');
});

test('release metrics use a matched comparator and fresh counts while retaining training history', async ({page})=>{
  const evidence=require('../../docs/results/final-release-completed.json');
  const old=require('../../docs/results/overnight-100k.json');
  let report={...old, release_evaluation:{
    test:evidence.final_audits.incumbent.metrics,
    baseline:evidence.final_audits.packaged.metrics,
    baseline_label:'Broad-training baseline',selection_states:8192,
  }};
  await page.route('**/api/sessions**',async route=>{
    const response=await route.fetch();
    const json=await response.json();
    if(json.model)json.model.report=report;
    await route.fulfill({response,json});
  });
  await page.goto('/');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#report-badge')).toContainText('16,384 TEST STATES');
  await expect(page.locator('#training-report')).toContainText('Broad-training baseline');
  await expect(page.locator('#training-report')).toContainText('92.8%');
  await expect(page.locator('#training-report')).toContainText('95.9%');
  await expect(page.locator('#training-report')).toContainText('8,192 separate selection states');
  await expect(page.locator('#training-report')).toContainText('Temperatures were retained');
  await expect(page.locator('#training-report')).not.toContainText('Warm start');
  report=old;
  await page.reload();
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.locator('#report-badge')).toContainText('5,000 TEST STATES');
  await expect(page.locator('#training-report')).toContainText('Warm start');
});
