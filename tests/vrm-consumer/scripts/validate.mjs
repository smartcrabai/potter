// Validates the built fixture .vrm files:
//  1. glTF validator: zero errors; every warning must match a reasoned allowlist entry;
//     every allowlist entry must match at least one warning in this run.
//  2. Official JSON schemas (ajv): VRM 1.0 extensions and mtoon materials, VRM 0.x extension.
// Exits nonzero on any failure.
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import { validateBytes } from "gltf-validator";
import { ensureSchemas } from "./fetch-schemas.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const fixtureDir = join(root, ".cache", "fixture");
const allowlistPath = join(root, "validate", "warning-allowlist.json");

const FIXTURES = [
  { file: "v1-full.vrm", kind: "v1" },
  { file: "v1-expression-lookat.vrm", kind: "v1" },
  { file: "v0.vrm", kind: "v0" },
];
const SEVERITY_WARNING = 1;
const MAX_PRINTED_ISSUES = 20;

// GLB container -> the JSON chunk as an object.
export function parseGlb(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2) {
    throw new Error("not a glTF 2.0 binary (GLB) file");
  }
  const length = view.getUint32(12, true);
  if (view.getUint32(16, true) !== 0x4e4f534a) throw new Error("first GLB chunk is not JSON");
  return JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + length)));
}

// Throws unless every entry has a code, a compiling pointer regex, and a non-empty reason.
export function validateAllowlistEntries(entries) {
  if (!Array.isArray(entries)) throw new Error("warning allowlist must be a JSON array");
  entries.forEach((entry, i) => {
    const where = `allowlist[${i}]`;
    if (typeof entry.code !== "string" || entry.code === "") throw new Error(`${where}: code is required`);
    if (typeof entry.pointer !== "string") throw new Error(`${where}: pointer regex is required`);
    if (typeof entry.reason !== "string" || entry.reason.trim() === "") throw new Error(`${where}: reason is required`);
    try {
      new RegExp(entry.pointer);
    } catch (err) {
      throw new Error(`${where}: pointer is not a valid regex: ${err.message}`);
    }
  });
}

// Splits warnings into those no entry covers, and counts hits per entry.
export function classifyWarnings(messages, entries) {
  const hits = entries.map(() => 0);
  const unallowed = [];
  for (const message of messages.filter((m) => m.severity === SEVERITY_WARNING)) {
    let matched = false;
    entries.forEach((entry, i) => {
      if (entry.code === message.code && new RegExp(entry.pointer).test(message.pointer ?? "")) {
        hits[i]++;
        matched = true;
      }
    });
    if (!matched) unallowed.push(message);
  }
  return { unallowed, hits };
}

// Schema checks for one parsed glTF: [{ schema, data, pointer, required }].
export function schemaTargets(gltf, kind) {
  const ext = gltf.extensions ?? {};
  if (kind === "v0") {
    return [{ schema: "vrm.schema.json", data: ext.VRM, pointer: "/extensions/VRM", required: true }];
  }
  const targets = [
    { schema: "VRMC_vrm.schema.json", data: ext.VRMC_vrm, pointer: "/extensions/VRMC_vrm", required: true },
  ];
  const spring = ext.VRMC_springBone;
  if (spring !== undefined) {
    targets.push({ schema: "VRMC_springBone.schema.json", data: spring, pointer: "/extensions/VRMC_springBone" });
    (spring.colliders ?? []).forEach((collider, i) => {
      const data = collider.extensions?.VRMC_springBone_extended_collider;
      if (data !== undefined) {
        targets.push({
          schema: "VRMC_springBone_extended_collider.schema.json",
          data,
          pointer: `/extensions/VRMC_springBone/colliders/${i}/extensions/VRMC_springBone_extended_collider`,
        });
      }
    });
  }
  (gltf.materials ?? []).forEach((material, i) => {
    const data = material.extensions?.VRMC_materials_mtoon;
    if (data !== undefined) {
      targets.push({
        schema: "VRMC_materials_mtoon.schema.json",
        data,
        pointer: `/materials/${i}/extensions/VRMC_materials_mtoon`,
      });
    }
  });
  return targets;
}

async function compileSchemas(paths) {
  // The official schemas are used as published. They carry glTF doc annotations (gltf_*), one file
  // declares draft-04 (not a bundled meta-schema), and some oneOf branches list `required` names that
  // are not in `properties`. Those are therefore not errors; the schemas are integrity-checked by lock.
  const ajv = new Ajv2020({ allErrors: true, strictSchema: false, strictRequired: false, validateSchema: false });
  for (const [name, path] of Object.entries(paths)) {
    ajv.addSchema(JSON.parse(await readFile(path, "utf8")), name);
  }
  return ajv;
}

function formatIssue(issue) {
  return `${issue.code} at ${issue.pointer ?? "(no pointer)"}: ${issue.message}`;
}

async function main() {
  const missing = FIXTURES.filter((f) => !existsSync(join(fixtureDir, f.file))).map((f) => f.file);
  if (missing.length > 0) {
    throw new Error(`missing fixture files in ${fixtureDir}: ${missing.join(", ")}; run node scripts/build-fixture.mjs`);
  }
  const allowlist = JSON.parse(await readFile(allowlistPath, "utf8"));
  validateAllowlistEntries(allowlist);
  const { paths } = await ensureSchemas();
  const ajv = await compileSchemas(paths);

  const failures = [];
  const warningCounts = new Map(); // code -> { total, allowlisted }
  const allHits = allowlist.map(() => 0);
  const fileResults = [];

  for (const { file, kind } of FIXTURES) {
    const problems = [];
    const bytes = new Uint8Array(await readFile(join(fixtureDir, file)));
    const report = await validateBytes(bytes, { maxIssues: 0 });
    const { numErrors, messages, truncated } = report.issues;
    if (truncated) problems.push("glTF validator output was truncated");
    if (numErrors > 0) {
      problems.push(`${numErrors} glTF validator error(s)`);
      for (const m of messages.filter((m) => m.severity === 0).slice(0, MAX_PRINTED_ISSUES)) {
        problems.push(`  error ${formatIssue(m)}`);
      }
    }

    const { unallowed, hits } = classifyWarnings(messages, allowlist);
    hits.forEach((n, i) => (allHits[i] += n));
    for (const m of messages.filter((m) => m.severity === SEVERITY_WARNING)) {
      const entry = warningCounts.get(m.code) ?? { total: 0, allowlisted: 0 };
      entry.total++;
      if (!unallowed.includes(m)) entry.allowlisted++;
      warningCounts.set(m.code, entry);
    }
    for (const m of unallowed.slice(0, MAX_PRINTED_ISSUES)) problems.push(`  unallowlisted warning ${formatIssue(m)}`);
    if (unallowed.length > MAX_PRINTED_ISSUES) problems.push(`  ...${unallowed.length - MAX_PRINTED_ISSUES} more unallowlisted warnings`);

    const gltf = parseGlb(bytes);
    for (const target of schemaTargets(gltf, kind)) {
      if (target.data === undefined) {
        if (target.required) problems.push(`${target.pointer} is missing`);
        continue;
      }
      const validate = ajv.getSchema(target.schema);
      if (!validate(target.data)) {
        for (const err of validate.errors) {
          problems.push(`  schema ${target.schema}: ${target.pointer}${err.instancePath}: ${err.message}`);
        }
      }
    }

    const warnings = messages.filter((m) => m.severity === SEVERITY_WARNING).length;
    if (problems.length === 0) {
      fileResults.push(`PASS ${file}: 0 errors, ${warnings} warnings (all allowlisted), schemas ok`);
    } else {
      fileResults.push(`FAIL ${file}: ${problems.length} problem(s)`);
      failures.push(...problems.map((p) => `${file}: ${p}`));
    }
  }

  const stale = allowlist.filter((_, i) => allHits[i] === 0);
  for (const entry of stale) failures.push(`stale allowlist entry (matched no warning): ${entry.code} ${entry.pointer}`);

  console.log("warnings by code (all fixtures):");
  for (const [code, { total, allowlisted }] of [...warningCounts].sort()) {
    console.log(`  ${code}: ${total} (allowlisted ${allowlisted})`);
  }
  console.log("allowlist entries (hits):");
  allowlist.forEach((entry, i) => console.log(`  ${entry.code} ${entry.pointer} -> ${allHits[i]}`));
  for (const line of fileResults) console.log(line);
  for (const line of failures) console.log(line);
  if (failures.length > 0) process.exitCode = 1;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err.message);
    process.exitCode = 1;
  });
}
