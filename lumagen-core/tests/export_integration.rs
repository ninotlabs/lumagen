//! Integration test: the export pipeline writes a real, complete bundle to disk (per-map files,
//! engine channel-pack, contact sheet, README, manifest) for a chosen engine preset.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::disallowed_methods)]

use lumagen_core::data::MapId;
use lumagen_core::export::{EngineConvention, ExportJob, export_material};
use lumagen_core::maps::TEX_SIZE;

fn solid(size: usize, v: u8) -> Vec<u8> {
    vec![v; size * size * 4]
}

/// Unique per test and per process — concurrent test runs must not share output paths.
/// Any leftover from a previous run is removed.
fn unique_temp_dir(test: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lumagen_export_{test}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// An `ExportJob` over `maps` with every field the tests don't vary held at a fixed default.
fn base_job(maps: [Option<Vec<u8>>; 8]) -> ExportJob {
    ExportJob {
        maps,
        map_sizes: [0; 8],
        src_size: TEX_SIZE,
        out_size: TEX_SIZE,
        material: "mat".into(),
        res: "1k".into(),
        pattern: "$material_$map_$res".into(),
        engine: EngineConvention::Generic,
        directx_normal: false,
        displacement_16bit_png: false,
        displacement_exr: false,
        displacement_height: None,
        seed: 1,
        prompt: String::new(),
        seam_margin: 0,
    }
}

#[test]
fn export_writes_full_bundle_for_unreal() {
    let mut maps: [Option<Vec<u8>>; 8] = Default::default();
    maps[MapId::Albedo.index()] = Some(solid(TEX_SIZE, 200));
    maps[MapId::Roughness.index()] = Some(solid(TEX_SIZE, 100));
    maps[MapId::Metallic.index()] = Some(solid(TEX_SIZE, 10));
    maps[MapId::Normal.index()] = Some(solid(TEX_SIZE, 128));
    maps[MapId::Ao.index()] = Some(solid(TEX_SIZE, 255));

    let job = ExportJob {
        material: "orbital_hull".into(),
        res: "4k".into(),
        engine: EngineConvention::Unreal,
        directx_normal: true,
        seed: 7,
        prompt: "painted hull".into(),
        ..base_job(maps)
    };

    let dir = unique_temp_dir("full_bundle_unreal");
    let written = export_material(&job, &dir).expect("export must succeed");

    assert!(written.iter().any(|p| p.to_string_lossy().contains("albedo")), "albedo missing");
    assert!(written.iter().any(|p| p.to_string_lossy().contains("orm")), "Unreal ORM pack missing");
    assert!(dir.join("contact_sheet.png").exists(), "contact sheet missing");
    assert!(dir.join("README.md").exists(), "README missing");

    let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest present");
    assert!(manifest.contains("\"material\": \"orbital_hull\""), "manifest material wrong: {manifest}");
    assert!(manifest.contains("\"engine\": \"Unreal\""), "manifest engine wrong");
    assert!(manifest.contains("DirectX"), "manifest normal convention not DirectX");
    assert!(manifest.contains("painted hull"), "manifest prompt provenance missing");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn export_derives_geometric_maps_and_writes_exr_for_blender() {
    // Blender preset defaults displacement to float EXR.
    let mut maps: [Option<Vec<u8>>; 8] = Default::default();
    maps[MapId::Albedo.index()] = Some(solid(TEX_SIZE, 180));
    maps[MapId::Displacement.index()] = Some(solid(TEX_SIZE, 90));

    let job = ExportJob {
        res: "2k".into(),
        pattern: "$material_$map".into(),
        engine: EngineConvention::Blender,
        displacement_exr: true,
        ..base_job(maps)
    };
    let dir = unique_temp_dir("exr_blender");
    let written = export_material(&job, &dir).expect("export");
    assert!(
        written.iter().any(|p| p.extension().and_then(|e| e.to_str()) == Some("exr")),
        "no EXR displacement written"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A hostile material name (loaded `.lumagen` documents carry it unsanitized) must neither
/// escape the chosen folder — `Path::join` replaces the base on an absolute component — nor
/// abort mid-export on Windows-invalid filename characters.
#[test]
fn export_with_hostile_material_name_stays_inside_out_dir() {
    let mut maps: [Option<Vec<u8>>; 8] = Default::default();
    maps[MapId::Albedo.index()] = Some(solid(TEX_SIZE, 200));

    let job = ExportJob {
        material: "..\\..\\C:\\evil<>:\"|?*hull".into(),
        seed: 3,
        ..base_job(maps)
    };
    let dir = unique_temp_dir("hostile_material");
    let written = export_material(&job, &dir).expect("export must survive a hostile material name");

    for p in &written {
        assert_eq!(p.parent(), Some(dir.as_path()), "file escaped the export dir: {}", p.display());
        assert!(p.exists(), "reported as written but missing: {}", p.display());
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        assert!(
            !name.contains(['/', '\\', '<', '>', ':', '"', '|', '?', '*']),
            "unsanitized filename written: {name}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
