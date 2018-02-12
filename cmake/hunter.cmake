option(HUNTER_ENABLED "Enable Hunter package manager support" OFF)
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.20.0.tar.gz"
    SHA1 "e94556ed41e5432997450bca7232db72a3b0d5ef"    
)
