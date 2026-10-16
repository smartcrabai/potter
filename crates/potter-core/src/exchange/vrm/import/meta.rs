//! VRM meta decoding. VRM 1.0 `VRMC_vrm.meta` stores typed fields directly; VRM 0.x `VRM.meta` stores
//! legacy permission names. Inverts `vrm1_meta`, `vrm0_meta` and the typed/legacy policy of
//! `validate_v0_meta_settings` in the exporter.

use serde_json::Value;

use crate::{
    error::Result,
    model::{
        AvatarCommercialUsage, AvatarCreditNotation, AvatarLegacyAllowedUser,
        AvatarLegacyBinaryPermission, AvatarLegacyLicenseName, AvatarLegacyMeta, AvatarMeta,
        AvatarModification, AvatarPermission, AvatarVersion, Id,
    },
};

use super::{Decoder, import_error};

/// The license URL VRM 1.0 requires; the exporter rejects any other `licenseUrl`.
const VRM1_LICENSE_URL: &str = "https://vrm.dev/licenses/1.0/";
/// Substituted when a required field is missing; each substitution records a `vrm.meta.<field>` loss.
const DEFAULT_NAME: &str = "Untitled";
const DEFAULT_AUTHOR: &str = "Unknown";

pub(super) fn decode(dec: &mut Decoder<'_>) -> Result<AvatarMeta> {
    // Potter writes the exact typed meta under `extras.potter.avatar_meta`, which VRM 0.x fields cannot hold.
    let saved = &dec.index.root["extras"]["potter"]["avatar_meta"];
    if !saved.is_null() {
        return serde_json::from_value(saved.clone())
            .map_err(|error| import_error(format!("glTF avatar metadata is invalid: {error}")));
    }
    match dec.version {
        AvatarVersion::V1 => decode_v1(dec),
        AvatarVersion::V0 => decode_v0(dec),
    }
}

/// Inverts `vrm1_meta`: every typed field is stored under its own key.
fn decode_v1(dec: &mut Decoder<'_>) -> Result<AvatarMeta> {
    let extension = dec.extension;
    let meta = &extension["meta"];
    let name = required_text(dec, meta, "name", "name", DEFAULT_NAME);
    let authors_raw = text_list(dec, meta, "authors", "authors");
    let authors = required_authors(dec, authors_raw);
    let license_url = required_text(dec, meta, "licenseUrl", "license_url", VRM1_LICENSE_URL);
    // `avatarPermission` is written by `vrm1_meta` from `AvatarPermission`; the VRM 1.0 default is onlyAuthor.
    let avatar_permission = enum_field(
        dec,
        meta,
        "avatarPermission",
        "avatar_permission",
        AvatarPermission::OnlyAuthor,
        |text| match text {
            "onlyAuthor" => Some(AvatarPermission::OnlyAuthor),
            "onlySeparatelyLicensedPerson" => Some(AvatarPermission::OnlySeparatelyLicensedPerson),
            "everyone" => Some(AvatarPermission::Everyone),
            _ => None,
        },
    );
    let allow_excessively_violent_usage = bool_field(
        dec,
        meta,
        "allowExcessivelyViolentUsage",
        "allow_excessively_violent_usage",
    );
    let allow_excessively_sexual_usage = bool_field(
        dec,
        meta,
        "allowExcessivelySexualUsage",
        "allow_excessively_sexual_usage",
    );
    let commercial_usage = enum_field(
        dec,
        meta,
        "commercialUsage",
        "commercial_usage",
        AvatarCommercialUsage::PersonalNonProfit,
        |text| match text {
            "personalNonProfit" => Some(AvatarCommercialUsage::PersonalNonProfit),
            "personalProfit" => Some(AvatarCommercialUsage::PersonalProfit),
            "corporation" => Some(AvatarCommercialUsage::Corporation),
            _ => None,
        },
    );
    let allow_political_or_religious_usage = bool_field(
        dec,
        meta,
        "allowPoliticalOrReligiousUsage",
        "allow_political_or_religious_usage",
    );
    let allow_antisocial_or_hate_usage = bool_field(
        dec,
        meta,
        "allowAntisocialOrHateUsage",
        "allow_antisocial_or_hate_usage",
    );
    let credit_notation = enum_field(
        dec,
        meta,
        "creditNotation",
        "credit_notation",
        AvatarCreditNotation::Required,
        |text| match text {
            "required" => Some(AvatarCreditNotation::Required),
            "unnecessary" => Some(AvatarCreditNotation::Unnecessary),
            _ => None,
        },
    );
    let allow_redistribution = bool_field(dec, meta, "allowRedistribution", "allow_redistribution");
    let modification = enum_field(
        dec,
        meta,
        "modification",
        "modification",
        AvatarModification::Prohibited,
        |text| match text {
            "prohibited" => Some(AvatarModification::Prohibited),
            "allowModification" => Some(AvatarModification::AllowModification),
            "allowModificationRedistribution" => {
                Some(AvatarModification::AllowModificationRedistribution)
            }
            _ => None,
        },
    );
    let version = text_field(dec, meta, "version", "version");
    let copyright_information =
        text_field(dec, meta, "copyrightInformation", "copyright_information");
    let contact_information = text_field(dec, meta, "contactInformation", "contact_information");
    let third_party_licenses = text_field(dec, meta, "thirdPartyLicenses", "third_party_licenses");
    let other_license_url = text_field(dec, meta, "otherLicenseUrl", "other_license_url");
    let references = text_list(dec, meta, "references", "references");
    let thumbnail = v1_thumbnail(dec, &meta["thumbnailImage"])?;
    Ok(AvatarMeta {
        name,
        version,
        authors,
        copyright_information,
        contact_information,
        references,
        third_party_licenses,
        thumbnail,
        license_url,
        avatar_permission,
        allow_excessively_violent_usage,
        allow_excessively_sexual_usage,
        commercial_usage,
        allow_political_or_religious_usage,
        allow_antisocial_or_hate_usage,
        credit_notation,
        allow_redistribution,
        modification,
        other_license_url,
        legacy: None,
    })
}

/// Inverts `vrm0_meta` and the typed/legacy policy of `validate_v0_meta_settings`.
///
/// VRM 0.x has no `licenseUrl`, `copyrightInformation`, `thirdPartyLicenses` or credit field, and its
/// permissions are binary legacy names. Each typed field is the inverse of the value the exporter derives
/// from it; `legacy` is set only where the exporter would otherwise derive a different value, so a Potter
/// meta without overrides reproduces exactly.
fn decode_v0(dec: &mut Decoder<'_>) -> Result<AvatarMeta> {
    let extension = dec.extension;
    let meta = &extension["meta"];
    // `title` is `legacy.title.unwrap_or(&meta.name)`; the decoded name is the title.
    let name = required_text(dec, meta, "title", "name", DEFAULT_NAME);
    // `author` is `legacy.author` or `authors.join("\n")`.
    let author = text_field(dec, meta, "author", "authors").unwrap_or_default();
    let authors = required_authors(dec, author.split('\n').map(str::to_owned).collect());
    let author_override =
        (!author.trim().is_empty() && authors.join("\n") != author).then_some(author);

    // `allowedUserName` is `expected_allowed_user(avatar_permission)`, a one-to-one mapping.
    let allowed_user = enum_field(
        dec,
        meta,
        "allowedUserName",
        "avatar_permission",
        AvatarLegacyAllowedUser::OnlyAuthor,
        parse_legacy_allowed_user,
    );
    let avatar_permission = match allowed_user {
        AvatarLegacyAllowedUser::OnlyAuthor => AvatarPermission::OnlyAuthor,
        AvatarLegacyAllowedUser::ExplicitlyLicensedPerson => {
            AvatarPermission::OnlySeparatelyLicensedPerson
        }
        AvatarLegacyAllowedUser::Everyone => AvatarPermission::Everyone,
    };
    // `violentUssageName`/`sexualUssageName` are `allow_excessively_*_usage` as Allow/Disallow.
    let violent = enum_field(
        dec,
        meta,
        "violentUssageName",
        "allow_excessively_violent_usage",
        AvatarLegacyBinaryPermission::Disallow,
        parse_binary_permission,
    );
    let sexual = enum_field(
        dec,
        meta,
        "sexualUssageName",
        "allow_excessively_sexual_usage",
        AvatarLegacyBinaryPermission::Disallow,
        parse_binary_permission,
    );
    let license = enum_field(
        dec,
        meta,
        "licenseName",
        "license_name",
        AvatarLegacyLicenseName::RedistributionProhibited,
        parse_legacy_license_name,
    );
    let commercial = enum_field(
        dec,
        meta,
        "commercialUssageName",
        "commercial_usage",
        AvatarLegacyBinaryPermission::Disallow,
        parse_binary_permission,
    );
    let other_permission_url = text_field(dec, meta, "otherPermissionUrl", "other_permission_url")
        .filter(|url| !url.trim().is_empty());
    let other_license_url = text_field(dec, meta, "otherLicenseUrl", "other_license_url")
        .filter(|url| !url.trim().is_empty());
    if license == AvatarLegacyLicenseName::Other && other_license_url.is_none() {
        dec.loss(
            "vrm.meta.other_license_url",
            "VRM 0.x licenseName `Other` has no otherLicenseUrl",
            "add otherLicenseUrl to the VRM file before importing",
        );
    }
    let version = text_field(dec, meta, "version", "version");
    let contact_information = text_field(dec, meta, "contactInformation", "contact_information");
    // `reference` is `legacy.reference` or `references.join("\n")`.
    let reference = text_field(dec, meta, "reference", "references").unwrap_or_default();
    let references = if reference.is_empty() {
        Vec::new()
    } else {
        reference.split('\n').map(str::to_owned).collect()
    };

    // Inverse of the license checks in `validate_v0_meta_settings`. Creative Commons licenses grant
    // redistribution; `Redistribution_Prohibited` and `Other` (terms live at otherLicenseUrl) do not.
    let creative_commons = !matches!(
        license,
        AvatarLegacyLicenseName::RedistributionProhibited | AvatarLegacyLicenseName::Other
    );
    // Share-alike licenses permit adapted material to be shared (`AllowModificationRedistribution`);
    // no-derivatives licenses permit private adaptation only (`AllowModification`).
    let modification = match license {
        AvatarLegacyLicenseName::Cc0
        | AvatarLegacyLicenseName::CcBy
        | AvatarLegacyLicenseName::CcBySa
        | AvatarLegacyLicenseName::CcByNc
        | AvatarLegacyLicenseName::CcByNcSa => AvatarModification::AllowModificationRedistribution,
        AvatarLegacyLicenseName::CcByNd | AvatarLegacyLicenseName::CcByNcNd => {
            AvatarModification::AllowModification
        }
        AvatarLegacyLicenseName::RedistributionProhibited | AvatarLegacyLicenseName::Other => {
            AvatarModification::Prohibited
        }
    };
    // CC0 waives attribution; the exporter rejects `Unnecessary`, so a CC0 file cannot be re-exported to VRM 0.x.
    let credit_notation = if license == AvatarLegacyLicenseName::Cc0 {
        AvatarCreditNotation::Unnecessary
    } else {
        AvatarCreditNotation::Required
    };
    // CC0, CC BY, CC BY-SA and CC BY-ND require `commercial_usage` corporation; other values follow the
    // binary `commercialUssageName` (Allow requires corporation, Disallow is personal non-profit).
    let corporation_license = matches!(
        license,
        AvatarLegacyLicenseName::Cc0
            | AvatarLegacyLicenseName::CcBy
            | AvatarLegacyLicenseName::CcBySa
            | AvatarLegacyLicenseName::CcByNd
    );
    let commercial_usage =
        if commercial == AvatarLegacyBinaryPermission::Allow || corporation_license {
            AvatarCommercialUsage::Corporation
        } else {
            AvatarCommercialUsage::PersonalNonProfit
        };
    let legacy = AvatarLegacyMeta {
        author: author_override,
        other_permission_url,
        // The exporter requires an explicit legacy commercial value whenever commercial_usage is not personal non-profit.
        commercial_usage: (commercial_usage != AvatarCommercialUsage::PersonalNonProfit)
            .then_some(commercial),
        license_name: (license != AvatarLegacyLicenseName::RedistributionProhibited)
            .then_some(license),
        ..AvatarLegacyMeta::default()
    };
    let legacy = (legacy != AvatarLegacyMeta::default()).then_some(legacy);
    let thumbnail = v0_thumbnail(dec, &meta["texture"])?;
    Ok(AvatarMeta {
        name,
        version,
        authors,
        copyright_information: None,
        contact_information,
        references,
        third_party_licenses: None,
        thumbnail,
        // VRM 0.x has no licenseUrl. The VRM 1.0 required value is used; the legal terms are carried by
        // legacy license_name and other_license_url.
        license_url: VRM1_LICENSE_URL.to_owned(),
        avatar_permission,
        allow_excessively_violent_usage: violent == AvatarLegacyBinaryPermission::Allow,
        allow_excessively_sexual_usage: sexual == AvatarLegacyBinaryPermission::Allow,
        commercial_usage,
        // VRM 0.x has no typed field; the exporter folds these into otherPermissionUrl. Defaulted to disallowed.
        allow_political_or_religious_usage: false,
        allow_antisocial_or_hate_usage: false,
        credit_notation,
        allow_redistribution: creative_commons,
        modification,
        other_license_url,
        legacy,
    })
}

/// Reads an optional string field. A present value of another JSON type records a loss.
fn text_field(dec: &mut Decoder<'_>, object: &Value, key: &str, field: &str) -> Option<String> {
    let value = &object[key];
    if value.is_null() {
        return None;
    }
    if let Some(text) = value.as_str() {
        return Some(text.to_owned());
    }
    dec.loss(
        &format!("vrm.meta.{field}"),
        format!("VRM meta `{key}` is not a string"),
        "correct the VRM file before importing",
    );
    None
}

/// Reads a required string field. A missing or blank value records a loss and uses `default`.
fn required_text(
    dec: &mut Decoder<'_>,
    object: &Value,
    key: &str,
    field: &str,
    default: &str,
) -> String {
    match text_field(dec, object, key, field) {
        Some(text) if !text.trim().is_empty() => text,
        _ => {
            dec.loss(
                &format!("vrm.meta.{field}"),
                format!("VRM meta `{key}` is missing or empty"),
                "add it to the VRM file before importing",
            );
            default.to_owned()
        }
    }
}

/// Reads an optional array of strings. Non-string entries record a loss and are dropped.
fn text_list(dec: &mut Decoder<'_>, object: &Value, key: &str, field: &str) -> Vec<String> {
    let value = &object[key];
    let Some(items) = value.as_array() else {
        if !value.is_null() {
            dec.loss(
                &format!("vrm.meta.{field}"),
                format!("VRM meta `{key}` is not an array"),
                "correct the VRM file before importing",
            );
        }
        return Vec::new();
    };
    let mut texts = Vec::with_capacity(items.len());
    let mut all_strings = true;
    for item in items {
        match item.as_str() {
            Some(text) => texts.push(text.to_owned()),
            None => all_strings = false,
        }
    }
    if !all_strings {
        dec.loss(
            &format!("vrm.meta.{field}"),
            format!("VRM meta `{key}` has non-string entries"),
            "correct the VRM file before importing",
        );
    }
    texts
}

/// Authors must be non-blank. Blank entries are dropped; an empty result uses `DEFAULT_AUTHOR`.
fn required_authors(dec: &mut Decoder<'_>, authors: Vec<String>) -> Vec<String> {
    let total = authors.len();
    let kept: Vec<String> = authors
        .into_iter()
        .filter(|author| !author.trim().is_empty())
        .collect();
    if kept.len() == total && !kept.is_empty() {
        return kept;
    }
    dec.loss(
        "vrm.meta.authors",
        "VRM meta authors are missing or blank",
        "add a non-empty author to the VRM file before importing",
    );
    if kept.is_empty() {
        vec![DEFAULT_AUTHOR.to_owned()]
    } else {
        kept
    }
}

/// Reads an enumerated string. A missing key uses `default`; an unrecognized value records a loss.
fn enum_field<T>(
    dec: &mut Decoder<'_>,
    object: &Value,
    key: &str,
    field: &str,
    default: T,
    parse: impl Fn(&str) -> Option<T>,
) -> T {
    let value = &object[key];
    if value.is_null() {
        return default;
    }
    let Some(parsed) = value.as_str().and_then(parse) else {
        dec.loss(
            &format!("vrm.meta.{field}"),
            format!("VRM meta `{key}` has an unsupported value"),
            "correct the VRM file before importing",
        );
        return default;
    };
    parsed
}

/// Reads a boolean flag. A missing key is `false`; a non-boolean value records a loss.
fn bool_field(dec: &mut Decoder<'_>, object: &Value, key: &str, field: &str) -> bool {
    match &object[key] {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        _ => {
            dec.loss(
                &format!("vrm.meta.{field}"),
                format!("VRM meta `{key}` is not a boolean"),
                "correct the VRM file before importing",
            );
            false
        }
    }
}

/// Inverts `vrm1_meta`'s `thumbnailImage`, a glTF image index.
fn v1_thumbnail(dec: &mut Decoder<'_>, value: &Value) -> Result<Option<Id>> {
    let Some(index) = value.as_u64().and_then(|index| usize::try_from(index).ok()) else {
        if !value.is_null() {
            dec.loss(
                "vrm.meta.thumbnail",
                "VRM thumbnailImage is not a glTF image index",
                "correct the VRM file before importing",
            );
        }
        return Ok(None);
    };
    dec.image_id(index).map(Some)
}

/// Inverts `vrm0_meta`'s `texture`, a glTF texture index; the thumbnail is that texture's source image.
fn v0_thumbnail(dec: &mut Decoder<'_>, value: &Value) -> Result<Option<Id>> {
    if value.is_null() {
        return Ok(None);
    }
    let source = value
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| dec.index.root["textures"].get(index))
        .and_then(|texture| texture["source"].as_u64())
        .and_then(|source| usize::try_from(source).ok());
    let Some(source) = source else {
        dec.loss(
            "vrm.meta.thumbnail",
            "VRM 0.x texture does not reference an image",
            "correct the texture reference in the VRM file",
        );
        return Ok(None);
    };
    dec.image_id(source).map(Some)
}

/// Inverts the exporter's `Disallow`/`Allow` names.
fn parse_binary_permission(text: &str) -> Option<AvatarLegacyBinaryPermission> {
    match text {
        "Disallow" => Some(AvatarLegacyBinaryPermission::Disallow),
        "Allow" => Some(AvatarLegacyBinaryPermission::Allow),
        _ => None,
    }
}

/// Inverts `expected_allowed_user`: the `allowedUserName` names written by `vrm0_meta`.
fn parse_legacy_allowed_user(text: &str) -> Option<AvatarLegacyAllowedUser> {
    match text {
        "OnlyAuthor" => Some(AvatarLegacyAllowedUser::OnlyAuthor),
        "ExplicitlyLicensedPerson" => Some(AvatarLegacyAllowedUser::ExplicitlyLicensedPerson),
        "Everyone" => Some(AvatarLegacyAllowedUser::Everyone),
        _ => None,
    }
}

/// Inverts `legacy_license_name`.
fn parse_legacy_license_name(text: &str) -> Option<AvatarLegacyLicenseName> {
    match text {
        "Redistribution_Prohibited" => Some(AvatarLegacyLicenseName::RedistributionProhibited),
        "CC0" => Some(AvatarLegacyLicenseName::Cc0),
        "CC_BY" => Some(AvatarLegacyLicenseName::CcBy),
        "CC_BY_NC" => Some(AvatarLegacyLicenseName::CcByNc),
        "CC_BY_SA" => Some(AvatarLegacyLicenseName::CcBySa),
        "CC_BY_NC_SA" => Some(AvatarLegacyLicenseName::CcByNcSa),
        "CC_BY_ND" => Some(AvatarLegacyLicenseName::CcByNd),
        "CC_BY_NC_ND" => Some(AvatarLegacyLicenseName::CcByNcNd),
        "Other" => Some(AvatarLegacyLicenseName::Other),
        _ => None,
    }
}
