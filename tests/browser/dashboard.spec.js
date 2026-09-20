const {test, expect} = require('@playwright/test');

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
