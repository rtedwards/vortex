// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::DecimalArray;
use crate::match_each_decimal_value_type;

pub(super) fn check_decimal_identical(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
) -> VortexResult<bool> {
    match_each_decimal_value_type!(lhs.values_type(), |T| {
        Ok(lhs.buffer::<T>() == rhs.buffer::<T>())
    })
}
