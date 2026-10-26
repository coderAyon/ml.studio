#define MyAppName "ML Studio"
#define MyAppVersion "0.4.0"
[Setup]
AppId={{45C1167B-7C9B-4D49-A850-33F06A25E531}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\ML Studio
DefaultGroupName=ML Studio
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=MLStudio-0.4.0-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
[Files]
Source: "..\dist\MLStudio-Research\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\ML Studio"; Filename: "{app}\MLStudio.exe"
Name: "{group}\Uninstall ML Studio"; Filename: "{uninstallexe}"
[Run]
Filename: "{app}\MLStudio.exe"; Description: "Launch ML Studio"; Flags: nowait postinstall skipifsilent
