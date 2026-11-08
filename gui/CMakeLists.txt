set(CMAKE_AUTOMOC ON)
qt_add_executable(xlamina WIN32
    main.cpp
    mainwindow.cpp mainwindow.h
    imageview.cpp imageview.h
    compare.cpp compare.h
    sweepdialog.cpp sweepdialog.h
    worker.cpp worker.h
    prefs.cpp prefs.h
    benchmark.cpp benchmark.h
    xlamina.rc)
# the window / taskbar icon, from the PNG sizes (PNG needs no image plugin)
qt_add_resources(xlamina "icons" PREFIX "/" BASE "icon" FILES
    icon/xlamina-16.png icon/xlamina-24.png icon/xlamina-32.png icon/xlamina-48.png
    icon/xlamina-64.png icon/xlamina-128.png icon/xlamina-256.png)
target_link_libraries(xlamina PRIVATE stackcore Qt6::Widgets)
target_include_directories(xlamina PRIVATE ${CMAKE_SOURCE_DIR}/cli)
set_target_properties(xlamina PROPERTIES OUTPUT_NAME "xLamina")
install(TARGETS xlamina RUNTIME DESTINATION .)
