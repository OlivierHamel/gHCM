

local bIsWindows    = os.get() == "windows"
-- \todo Add OCL for NV? Then again, they give a damn about OCL support so why bother.
local strOclIncDir  = bIsWindows and "$(AMDAPPSDKROOT)/include"     or "$(HOME)/AMDAPPSDK/include"
local strOclLibDir  = bIsWindows and "$(AMDAPPSDKROOT)/lib/x86_64"  or "$(HOME)/AMDAPPSDK/lib/x86_64"

workspace "gHCM"
   configurations { "debug", "release" }


project "gHCM"
  architecture  "x86_64"
  kind          "ConsoleApp"
  language      "C++"
  targetdir     ".build/%{cfg.buildcfg}"
  objdir        ".build/%{cfg.buildcfg}/obj"

  includedirs { "lib", "src", strOclIncDir }
  libdirs     { strOclLibDir  }
  links       { "OpenCL"      } -- either a project's name, or the name of a sys lib w/o ext

  pchheader "pch.h"
  pchsource "src/pch.cpp"

  forceincludes "pch.h"

  files       { "src/**.h", "src/**.cpp", "lib/docopt/docopt.cpp" }
  flags       { }

  filter { "system:linux" }
    buildoptions { "-std=c++14", "-stdlib=libc++" }
    linkoptions { "-v", "-stdlib=libc++" }
    links { "c++" }

  filter "configurations:debug"
    defines   { "DEBUG"   }
    symbols   "On"

  filter "configurations:release"
    defines   { "NDEBUG"  }
    symbols   "On"
    optimize  "On"

