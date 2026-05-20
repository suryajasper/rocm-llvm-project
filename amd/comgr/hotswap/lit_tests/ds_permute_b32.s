; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 \
; RUN:     --emit-ir=ds_permute_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; DS_PERMUTE_B32 must lower to `llvm.amdgcn.ds.permute(index, src)`.
; This is the forward (scatter) permute — semantically the inverse of
; ds_bpermute_b32 (gather).  Both exist on gfx950.
;
; The handler applies the same wave32->wave64 selector rebasing that
; the DS_BPERMUTE_B32 handler does.

; CHECK-LABEL: define amdgpu_kernel void @ds_permute_b32_kernel(
; CHECK:      %perm = call i32 @llvm.amdgcn.ds.permute(i32 %{{[^,]+}}, i32 %{{[^,]+}})
; CHECK: declare {{.*}}i32 @llvm.amdgcn.ds.permute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_permute_b32_kernel
	.p2align	8
	.type	ds_permute_b32_kernel,@function
ds_permute_b32_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_and_b32 v1, v0, 31
	v_lshlrev_b32 v2, 2, v1
	v_add_nc_u32 v2, v2, 64
	global_load_b32 v3, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	ds_permute_b32 v4, v2, v3
	s_wait_dscnt 0x0
	v_add_nc_u32 v3, v3, v4
	global_store_b32 v0, v3, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_permute_b32_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           ds_permute_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_permute_b32_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
