idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

set(src_dirs        .)

set(include_dirs    .)

set(requires        esp-dl)

set(packed_model ${BUILD_DIR}/espdl_models/custom_detect.espdl)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

if (NOT CONFIG_ESPDET_DETECT_MODEL_IN_SDCARD)
    if (IDF_TARGET STREQUAL "esp32s3")
        set(models_dir ${COMPONENT_DIR}/models/s3)
    elseif (IDF_TARGET STREQUAL "esp32p4")
        set(models_dir ${COMPONENT_DIR}/models/p4)
    endif()

    file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
    set(models)
    if(CONFIG_ESPDET_PICO_imgsz_imgsz_CUSTOM)
        list(APPEND models ${models_dir}/espdet_pico_imgsz_imgsz_custom.espdl)
    endif()


    set(pack_model_exe ${espdl_dir}/fbs_loader/pack_espdl_models.py)
    add_custom_command(
        OUTPUT ${packed_model}
        COMMENT "Move and Pack models..."
        COMMAND python ${pack_model_exe} --model_path ${models} --out_file ${packed_model}
        DEPENDS ${models}
        VERBATIM)

    if(CONFIG_ESPDET_DETECT_MODEL_IN_FLASH_RODATA)
        target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
    endif()

    if(CONFIG_ESPDET_DETECT_MODEL_IN_FLASH_PARTITION)
        add_custom_target(ESPDET_DETECT_MODEL ALL DEPENDS ${packed_model})
        add_dependencies(flash ESPDET_DETECT_MODEL)
        esptool_py_flash_to_partition(flash "espdet_det" ${packed_model})
    endif()
endif()
