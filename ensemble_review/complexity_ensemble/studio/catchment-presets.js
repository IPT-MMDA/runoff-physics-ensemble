// Apply a preset only if the dataset, source and manual selection still match.
export async function selectHydropowerCohort(api, context, apply) {
  const before = context(), signature = JSON.stringify(before);
  if (!before.ready) throw new Error("Wait for the selected catchment dataset to finish loading.");
  let preset;
  try {
    preset = await api(`/datasets/${encodeURIComponent(before.dataset)}/hydropower-cohort?mode=${encodeURIComponent(before.mode)}`);
  } catch (error) {
    if (signature !== JSON.stringify(context())) return null;
    if (/^(not found|request failed \(404\))$/i.test(error.message.trim())) {
      throw new Error("Restart the Studio/API server to load the Hydropower 50 preset, then refresh this page.");
    }
    throw error;
  }
  if (signature !== JSON.stringify(context())) return null;
  const ids = preset.basin_ids;
  if (!preset.available || !Array.isArray(ids) || ids.length !== 50 || new Set(ids).size !== 50) {
    const exclusions = Object.entries(preset.unavailable_basins || {}).map(([id, reason]) => `${id}: ${reason}`).join("; ");
    throw new Error(`The fixed Hydropower 50 preset is unavailable for this dataset/source. ${exclusions}`.trim());
  }
  apply(ids);
  return preset;
}
