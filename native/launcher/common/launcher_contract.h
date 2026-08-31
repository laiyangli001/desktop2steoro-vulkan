#pragma once

// Stable bootstrap status values shared by native launchers and future licensing code.
enum class D2SLauncherStatus {
    Ok = 0,
    RuntimeMissing = 2,
    DisplayUnavailable = 3,
    SplashLoadFailed = 4,
    ProcessStartFailed = 5,
    ChildExitedBeforeReady = 6,
    ReadyTimeout = 7,
    LicenseInvalid = 20,
    LicenseCheckFailed = 21,
};

// Future licensing adapters must return a status and never embed credentials here.
using D2SLicenseCheck = D2SLauncherStatus (*)();
