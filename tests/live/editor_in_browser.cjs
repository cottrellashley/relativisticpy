// Drive the editor in a marimo notebook with Playwright, in Chromium, and print what was
// seen, as JSON, for tests/live/test_browser.py to judge.
//
//   node editor_in_browser.cjs URL [CHROMIUM ...]
//
// Playwright is found as Node finds modules, so NODE_PATH may name where it is installed.
// The browser is Playwright's own, or else the first of the executables given that starts.

const { chromium } = require("playwright");

async function launch(candidates) {
  const tried = [];
  for (const executablePath of [undefined, ...candidates]) {
    try {
      return await chromium.launch({ executablePath, headless: true });
    } catch (error) {
      tried.push(`${executablePath || "Playwright's Chromium"}: ${error.message.split("\n")[0]}`);
    }
  }
  throw new Error(`no Chromium starts: ${tried.join("; ")}`);
}

async function main() {
  const [url, ...candidates] = process.argv.slice(2);
  let browser;
  try {
    browser = await launch(candidates);
  } catch (error) {
    console.log(JSON.stringify({ skip: error.message }));
    return;
  }
  const seen = { errors: [] };
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  page.on("pageerror", (error) => seen.errors.push(error.message));
  const editor = page.locator(".rp-live");
  const statements = editor.locator(".rp-statement");
  const input = editor.locator("textarea");
  const latex = (scope) =>
    scope.locator('annotation[encoding="application/x-tex"]').allTextContents();

  async function settled(count) {
    await statements.nth(count - 1).waitFor({ timeout: 90000 });
    await page.locator(".rp-live.rp-running").waitFor({ state: "detached", timeout: 90000 });
  }

  try {
    await page.goto(url);
    await settled(1);
    seen.first = await latex(statements.nth(0).locator(".rp-value"));

    // Type more, as a person types it, and wait for it to be computed.
    await input.click();
    await page.keyboard.press("Control+End");
    await page.keyboard.type("\nK := R_{abcd} R^{abcd}\nK r^6\n\\frca{1}{2}", { delay: 15 });
    await settled(4);
    seen.statements = await statements.count();
    seen.written = await latex(editor.locator(".rp-written"));
    seen.kretschmann = await latex(statements.nth(2).locator(".rp-value"));
    seen.definition = await statements.nth(1).locator(".rp-said").first().textContent();
    seen.error = await statements.nth(3).locator(".rp-headline").textContent();
    seen.marked = await editor.locator(".rp-backdrop .rp-mark").allTextContents();
    seen.math = await editor.locator(".rp-results math").count();

    // The error under the pointer.
    const mark = await editor.locator(".rp-mark").first().boundingBox();
    await page.mouse.move(mark.x + 4, mark.y + mark.height / 2);
    await editor.locator(".rp-tip").waitFor({ state: "visible", timeout: 5000 });
    seen.tip = await editor.locator(".rp-tip").textContent();

    // A part of a statement, selected, then computed from the menu a right click on it
    // opens. Where the selected text is drawn is found in the copy of the text behind it.
    const source = await input.inputValue();
    const start = source.indexOf("R_{abcd} R^{abcd}");
    await input.evaluate((node, range) => {
      node.focus();
      node.setSelectionRange(range[0], range[1]);
    }, [start, start + "R_{abcd} R^{abcd}".length]);
    const at = await editor.locator(".rp-backdrop").evaluate((backdrop, offset) => {
      const walker = document.createTreeWalker(backdrop, NodeFilter.SHOW_TEXT);
      let passed = 0;
      while (walker.nextNode()) {
        const text = walker.currentNode;
        if (passed + text.length > offset) {
          const range = document.createRange();
          range.setStart(text, offset - passed + 2);
          range.setEnd(text, offset - passed + 3);
          const box = range.getBoundingClientRect();
          return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
        }
        passed += text.length;
      }
      return null;
    }, start);
    await page.mouse.click(at.x, at.y, { button: "right" });
    const menu = editor.locator(".rp-menu");
    await menu.waitFor({ state: "visible", timeout: 5000 });
    seen.menu = await menu.locator("button").allTextContents();
    await menu.locator('button[data-action="simplify"]').click();
    const answer = editor.locator(".rp-answer");
    await answer.first().waitFor({ timeout: 90000 });
    seen.answer = {
      asked: await answer.first().locator(".rp-action").textContent(),
      part: await latex(answer.first().locator(".rp-part")),
      values: await latex(answer.first().locator(".rp-value")),
      under: await statements.nth(1).locator(".rp-answer").count(),
    };

    // A result has a menu of its own.
    await statements.nth(2).locator(".rp-value").click({ button: "right" });
    await menu.waitFor({ state: "visible", timeout: 5000 });
    seen.resultMenu = await menu.locator("button").allTextContents();
    await page.keyboard.press("Escape");

    // Another cell of the notebook reads the editor's statements, and runs again.
    const counted = page.getByText("statements: 4");
    await counted.waitFor({ timeout: 30000 });
    seen.reacted = await counted.count();

    // show, typeset by marimo itself.
    seen.shown = await page.locator(".rp-shown .rp-step").count();
    seen.shownError = await page.locator(".rp-shown .rp-error").textContent();
  } catch (error) {
    seen.failure = error.message;
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(seen));
}

main();
