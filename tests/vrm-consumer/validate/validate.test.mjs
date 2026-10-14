import { test } from "node:test";
import assert from "node:assert/strict";
import { classifyWarnings, parseGlb, schemaTargets, validateAllowlistEntries } from "../scripts/validate.mjs";

const warning = (code, pointer) => ({ code, pointer, severity: 1, message: "" });

test("parseGlb returns the JSON chunk", () => {
  const json = Buffer.from('{"asset":{"version":"2.0"}}  ');
  const head = Buffer.alloc(20);
  head.write("glTF", 0);
  head.writeUInt32LE(2, 4);
  head.writeUInt32LE(20 + json.length, 8);
  head.writeUInt32LE(json.length, 12);
  head.write("JSON", 16);
  assert.deepEqual(parseGlb(new Uint8Array(Buffer.concat([head, json]))), { asset: { version: "2.0" } });
  assert.throws(() => parseGlb(new Uint8Array(20)), /not a glTF 2.0 binary/);
});

test("classifyWarnings: only warnings count, pointer regex is applied, hits are per entry", () => {
  const entries = [{ code: "A", pointer: "^/nodes/0$", reason: "r" }];
  const messages = [
    warning("A", "/nodes/0"),
    warning("A", "/nodes/10"),
    warning("B", "/nodes/0"),
    { code: "B", pointer: "/x", severity: 0, message: "" },
  ];
  const { unallowed, hits } = classifyWarnings(messages, entries);
  assert.deepEqual(hits, [1]);
  assert.deepEqual(unallowed.map((m) => `${m.code} ${m.pointer}`), ["A /nodes/10", "B /nodes/0"]);
});

test("validateAllowlistEntries rejects missing or empty reasons and bad regexes", () => {
  assert.doesNotThrow(() => validateAllowlistEntries([{ code: "A", pointer: "^/a$", reason: "because" }]));
  assert.throws(() => validateAllowlistEntries([{ code: "A", pointer: "^/a$", reason: "  " }]), /reason is required/);
  assert.throws(() => validateAllowlistEntries([{ code: "A", pointer: "(", reason: "x" }]), /not a valid regex/);
  assert.throws(() => validateAllowlistEntries([{ code: "", pointer: "", reason: "x" }]), /code is required/);
});

test("schemaTargets selects VRM 1.0 extensions, extended colliders, and mtoon materials", () => {
  const gltf = {
    extensions: {
      VRMC_vrm: { specVersion: "1.0" },
      VRMC_springBone: {
        colliders: [{ extensions: { VRMC_springBone_extended_collider: { specVersion: "1.0" } } }, {}],
      },
    },
    materials: [{ extensions: { VRMC_materials_mtoon: {} } }, {}],
  };
  assert.deepEqual(
    schemaTargets(gltf, "v1").map((t) => [t.schema, t.pointer]),
    [
      ["VRMC_vrm.schema.json", "/extensions/VRMC_vrm"],
      ["VRMC_springBone.schema.json", "/extensions/VRMC_springBone"],
      [
        "VRMC_springBone_extended_collider.schema.json",
        "/extensions/VRMC_springBone/colliders/0/extensions/VRMC_springBone_extended_collider",
      ],
      ["VRMC_materials_mtoon.schema.json", "/materials/0/extensions/VRMC_materials_mtoon"],
    ],
  );
  assert.deepEqual(schemaTargets({ extensions: { VRM: {} } }, "v0").map((t) => t.schema), ["vrm.schema.json"]);
});
