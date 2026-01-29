import test from "node:test";
import assert from "node:assert/strict";
import { buildSessionUpdate } from "../session.js";

test("buildSessionUpdate sets tools only when vector store provided", () => {
  const noTools = buildSessionUpdate({ voice: "ermil", vectorStoreId: "" });
  assert.equal(Array.isArray(noTools.session.tools), true);
  assert.equal(noTools.session.tools.length, 0);

  const withTools = buildSessionUpdate({ voice: "ermil", vectorStoreId: "vs" });
  assert.equal(withTools.session.tools.length, 1);
});
