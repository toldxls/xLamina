add_executable(stackcli stackcli.cpp stackcli.rc utf8.manifest)
target_link_libraries(stackcli PRIVATE stackcore)
stackcore_copy_dlls(stackcli)
